from __future__ import absolute_import
import logging
import math
import pickle
import warnings
import os
import numpy
from mxnet.optimizer import Optimizer
from mxnet.ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
                           multi_sum_sq, multi_lars, norm as NDnorm)
import mxnet as mx
from mxnet import nd
from mxnet.ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                           mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                           signsgd_update, signum_update, nag_mom_update, mp_nag_mom_update,
                           multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update,
                           multi_mp_sgd_mom_update, preloaded_multi_sgd_update,
                           preloaded_multi_sgd_mom_update, preloaded_multi_mp_sgd_update,
                           preloaded_multi_mp_sgd_mom_update, lamb_update_phase1, lamb_update_phase2,
                           mp_lamb_update_phase1, mp_lamb_update_phase2)


def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


class LARS(Optimizer):
    """the LARS optimizer from 'Large Batch Training of Convolution Networks' \
    (https://arxiv.org/abs/1708.03888)

    Behave mostly like SGD with momentum and weight decay but is scaling \
    adaptively the learning for each layer (except bias and batch norm parameters):
    w_norm = L2norm(weights)
    g_norm = L2norm(gradients)
    if w_norm > 0 and g_norm > 0:
        lr_layer = lr * lr_mult * eta * w_norm / (g_norm + weight_decay * w_norm + eps)
    else:
        lr_layer = lr * lr_mult

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    lars_eta : float, optional
        LARS coefficient used to scale the learning rate. Default set to 0.001.
    lars_epsilon : float, optional
        Optional epsilon in case of very small gradients. Default set to 0.
    momentum_correction : bool, optional
        If True scale momentum w.r.t global learning rate change (with an lr_scheduler) \
        as indicated in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour` \
        (https://arxiv.org/pdf/1706.02677.pdf)
        Default set to True.
    """

    def __init__(self, momentum=0.0, lazy_update=True, eta=0.001, eps=1e-9, **kwargs):
        super(LARS, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.eta = eta
        self.eps = eps
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None

    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.cur_lr is not None:
            self.last_lr = self.cur_lr

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if self.cur_lr is None:
            self.last_lr = lr
        self.cur_lr = lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
                lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')

            if not is_weight:
                self.wd_mult[n] = 0.0

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    # def _l2norm(self, v, rescale=False):
    #     """L2 Norm implementation"""
    #     v = v.astype('float32')
    #     if rescale:
    #         v *= self.rescale_grad
    #     norm = NDnorm(v).asnumpy()[0]
    #     return norm

    # def _get_lars(self, i, weight, g, lr, wd):
    #     """Returns a scaling factor for the learning rate for this layer"""
    #     name = self.idx2name[i] if i in self.idx2name else str(i)
    #     if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
    #         return lr
    #
    #     w_norm = self._l2norm(weight)
    #     g_norm = self._l2norm(g, rescale=True)
    #
    #     if w_norm > 0.0 and g_norm > 0.0:
    #         lars = self.eta * w_norm/(g_norm + wd * w_norm + self.eps)
    #     else:
    #         lars = 1.0
    #     return lars * lr

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert (isinstance(weight, NDArray))
            assert (isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            nb_params = len(indices)
            names = [self.idx2name[i] if i in self.idx2name else str(i) for i in indices]
            lars_idx = [i for i in range(nb_params) if
                        not (names[i].endswith('gamma') or names[i].endswith('beta') or
                             names[i].endswith('bias'))]
            # nb_lars = len(lars_idx)
            no_lars_idx = [i for i in range(nb_params) if
                           (names[i].endswith('gamma') or names[i].endswith('beta') or
                            names[i].endswith('bias'))]
            cur_ctx = weights[0].context
            full_idx = lars_idx + no_lars_idx
            new_lrs = array([lrs[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
            new_wds = array([wds[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
            new_weights = [weights[i] for i in full_idx]
            new_grads = [grads[i] for i in full_idx]
            new_states = [states[i] for i in full_idx]
            # Scale learning rate
            # This is very important
            new_lrs = new_lrs * self.eta
            #
            clip_num = 20
            lars = mx.nd.ones_like(new_lrs)
            w_sum_sq = multi_sum_sq(*new_weights, num_arrays=len(new_weights))
            g_sum_sq = multi_sum_sq(*new_grads, num_arrays=len(new_grads))

            multi_lars(lars, w_sum_sq, g_sum_sq, new_wds,
                       eta=1.0, eps=self.eps, rescale_grad=self.rescale_grad,
                       out=lars)
            mx.nd.clip(lars, a_min=1 / clip_num, a_max=clip_num, out=lars)
            new_lrs = new_lrs * lars

            # Same than usual using preloaded sgd functions
            sidx = 0
            while sidx < len(indices):
                eidx = sidx + len(new_weights[sidx:sidx + self.aggregate_num])
                if not multi_precision:
                    if self.momentum > 0:
                        preloaded_multi_sgd_mom_update(
                            *(_flatten_list(zip(new_weights[sidx:eidx],
                                                new_grads[sidx:eidx],
                                                new_states[sidx:eidx])) +
                              [new_lrs[sidx:eidx], new_wds[sidx:eidx]]),
                            out=new_weights[sidx:eidx],
                            num_weights=len(new_weights[sidx:eidx]),
                            **kwargs)
                    else:
                        preloaded_multi_sgd_update(
                            *(_flatten_list(zip(new_weights[sidx:eidx],
                                                new_grads[sidx:eidx])) +
                              [new_lrs[sidx:eidx], new_wds[sidx:eidx]]),
                            out=new_weights[sidx:eidx],
                            num_weights=len(new_weights[sidx:eidx]),
                            **kwargs)
                else:
                    if self.momentum > 0:
                        preloaded_multi_mp_sgd_mom_update(
                            *(_flatten_list(zip(new_weights[sidx:eidx],
                                                new_grads[sidx:eidx],
                                                *zip(*new_states[sidx:eidx]))) +
                              [new_lrs[sidx:eidx], new_wds[sidx:eidx]]),
                            out=new_weights[sidx:eidx],
                            num_weights=len(new_weights[sidx:eidx]),
                            **kwargs)
                    else:
                        preloaded_multi_mp_sgd_update(
                            *(_flatten_list(zip(new_weights[sidx:eidx],
                                                new_grads[sidx:eidx],
                                                list(zip(*new_states[sidx:eidx]))[1])) +
                              [new_lrs[sidx:eidx], new_wds[sidx:eidx]]),
                            out=new_weights[sidx:eidx],
                            num_weights=len(new_weights[sidx:eidx]),
                            **kwargs)
                sidx += self.aggregate_num
        else:
            raise ValueError

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)
