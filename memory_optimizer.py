from mxnet import nd
from mxnet.ndarray import array
import mxnet as mx


def norm(v):
    return nd.sqrt(nd.sum(v.reshape(-1) ** 2))


def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


class MemoryBankSGDOptimizer(object):
    def __init__(self, lr_scheduler, rescale_grad):
        self.lr_scheduler = lr_scheduler
        self.rescale_grad = rescale_grad
        self.momentum = 0.9
        self.wd = 5e-4

    def update(self, weight, grad, state, learning_rate):
        lr = learning_rate
        # do the regular sgd update flow
        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if state is not None:
            nd.sgd_mom_update(weight, grad, state, out=weight, lr=lr, wd=self.wd, **kwargs)
        else:
            raise ValueError


class MemoryBankLarsOptimizer(object):
    def __init__(self, lr_scheduler, rescale_grad, local_rank,
                 lars_clip=100, clip_gradient=None, summary_writer=None):
        self.lr_scheduler = lr_scheduler
        self.rescale_grad = rescale_grad
        self.clip_gradient = clip_gradient
        self.lars_clip = lars_clip
        self.gpu = mx.gpu(local_rank)
        self.momentum = 0.9
        self.wd = array([5e-4], ctx=self.gpu, dtype='float32')
        self.lars_coef = 0.001

    def get_lars(self, weight: nd.NDArray, grad: nd.NDArray):
        """ Lars scaling
        """
        norm_weight = norm(weight)
        norm_grad = norm(grad)
        norm_grad = norm_grad * self.rescale_grad
        # layer-wise adaptive rate scaling
        lars = norm_weight / (norm_grad + self.wd * norm_weight + 1e-9)
        # clip lars
        lars = nd.clip(lars, a_min=1.0 / self.lars_clip, a_max=self.lars_clip)
        return lars

    def update(self, weight, grad, state, learning_rate):
        """Updates the given parameter using the corresponding gradient and states.
        Parameters
        ----------
        weight: NDArray
            The parameter to be updated.
        grad: NDArray
            The gradient of the objective with respect to this parameter.
        state: any obj
            The state
        learning_rate: float
            Current learning rate.
        """
        assert isinstance(weight, nd.NDArray)

        new_lrs = array([learning_rate], ctx=self.gpu, dtype='float32')
        new_wds = self.wd
        new_weights = [weight]
        new_grads = [grad]
        new_states = [state]

        # lars
        lr_scale = self.get_lars(weight, grad)
        #
        new_lrs = new_lrs * self.lars_coef * lr_scale

        # do the regular sgd update flow
        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if state is not None:
            nd.preloaded_multi_sgd_mom_update(
                *(_flatten_list(zip(new_weights, new_grads, new_states)) +
                  [new_lrs, new_wds]),
                out=new_weights,
                num_weights=len(new_weights),
                **kwargs)

    def __call__(self):
        raise ValueError
