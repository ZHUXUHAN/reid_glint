import logging
import warnings
from collections import namedtuple

import horovod.mxnet as hvd
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from mxnet import ndarray
from mxnet.context import cpu
from mxnet.initializer import Uniform
from mxnet.module.module import Module
from horovod_mxnet.dist_opt import DistributedOptimizer
from optimizer import LARS
from mxnet.optimizer import SGD


class MemoryModuleGlint(object):
    def __init__(self, symbol, bn_symbol, batch_size, fc7_model, size,
                 rank, local_rank, memory_bank_list, memory_optimizer,
                 backbone_grad_rescale, memory_lr_scale_list,
                 embedding_size=512, head_num=1, logger=logging, ):
        # configure horovod
        self.memory_lr_scale_list = memory_lr_scale_list
        self.size = size
        self.rank = rank
        self.local_rank = local_rank
        self.gpu = mx.gpu(self.local_rank)
        self.cpu = mx.cpu()                                     # `device_id` is not needed for CPU.
        self.nd_cache = {}
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_update = 0

        self.batch_end_param = namedtuple(
            'batch_end_param',
            ['loss_list', 'num_epoch_list', 'epoch', 'num_update'])

        self.symbol = symbol
        # self.bn_symbol = bn_symbol
        #
        self.logger = logger
        self.backbone_module = Module(self.symbol,    ['data'], ['softmax_label'], logger=self.logger, context=self.gpu)
        # self.bn_module       = Module(self.bn_symbol, ['data'], None, logger=self.logger, context=self.gpu)
        self.head_num = head_num
        self.memory_bank_list = memory_bank_list
        self.memory_optimizer = memory_optimizer
        self.memory_lr = None
        self.loss_cache = None
        self.grad_cache = None

        assert isinstance(self.memory_bank_list, list)

        # init
        self.fc7_model = fc7_model

        # fp16
        self.backbone_grad_rescale = backbone_grad_rescale

        self.binded = False
        self.for_training = False
        self.inputs_need_grad = False
        self.params_initialized = False
        self.optimizer_initialized = False
        self._total_exec_bytes = 0

        self.global_label = None

    def forward_backward(self, data_batch):
        total_feature, total_label = self.forward(data_batch, is_train=True)
        self.backward_all(total_feature, total_label)

    @staticmethod
    def sync_aux_params(params):
        pass

    @staticmethod
    def broadcast_parameters(params):
        rank_0_dict = {}
        # Run broadcasts.
        for key, tensor in params.items():
            rank_0_dict[key] = hvd.broadcast(tensor, 0, key)
        return rank_0_dict

    @staticmethod
    def combine(data_batches):
        assert isinstance(data_batches, list), "data_batches must be a list."
        length = len(data_batches)
        total_data = [data_batches[0].data[0].reshape(0, -1)]
        total_label = [data_batches[0].label[0].reshape(0, 1)]
        data_shape = data_batches[0].data[0].shape
        if length > 1:
            for i in range(1, length):
                assert data_batches[i].data[0].shape[0] == data_batches[0].data[0].shape[0]
                total_data.append(data_batches[i].data[0].reshape(0, -1))
                total_label.append(data_batches[i].label[0].reshape(0, 1))
        # shuffle
        total_data = mx.nd.concat(*total_data, dim=1)
        total_data = total_data.reshape(-1, data_shape[1], data_shape[2], data_shape[3])
        total_label = mx.nd.concat(*total_label, dim=1)
        total_label = total_label.reshape(-1)
        return mx.io.DataBatch([total_data], [total_label])

    def fit(self, train_data_list, optimizer_params, batch_end_callback=None, kvstore='local',
            initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None):

        assert num_epoch is not None, 'please specify number of epochs'
        assert arg_params is None and aux_params is None

        provide_data_list = []
        provide_label_list = []
        for td in train_data_list:
            provide_data_list.append(td.provide_data)
            provide_label_list.append(td.provide_label)

        self.bind(data_shapes_list=provide_data_list, label_shapes_list=provide_label_list,
                  for_training=True)

        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(optimizer_params=optimizer_params)

        _arg_params, _aux_params = self.backbone_module.get_params()
        _arg_params_rank_0 = self.broadcast_parameters(_arg_params)
        _aux_params_rank_0 = self.broadcast_parameters(_aux_params)
        self.backbone_module.set_params(_arg_params_rank_0, _aux_params_rank_0)
        data_end_id = 0
        ################################################################################
        # training loop
        ################################################################################
        num_epoch_list = [0] * self.head_num
        for epoch in range(begin_epoch, num_epoch):
            nbatch = 0
            end_of_batch = False
            data_iter_list = []
            for i in range(self.head_num):
                train_data_list[i].reset()
                data_iter_list.append(iter(train_data_list[i]))
            next_data_batch_list = []
            for i in range(self.head_num):
                next_data_batch_list.append(next(data_iter_list[i]))
            while not end_of_batch:
                data_batch_list = next_data_batch_list
                data_batch = self.combine(data_batch_list)

                self.forward_backward(data_batch)
                self.update()
                assert not isinstance(data_batch, list)

                for i in range(self.head_num):
                    try:
                        next_data_batch_list[i] = next(data_iter_list[i])
                        self.prepare(next_data_batch_list[i], sparse_row_id_fn=None)
                    except StopIteration:
                        num_epoch_list[i] += 1
                        data_end_id += 1
                        if data_end_id != self.head_num:
                            train_data_list[i].reset()
                            data_iter_list[i] = iter(train_data_list[i])
                            next_data_batch_list[i] = next(data_iter_list[i])
                            logging.info('reset dataset_%d' % i)

                if batch_end_callback is not None:
                    batch_end_params = self.batch_end_param(
                        loss_list=self.loss_cache,
                        epoch=epoch,
                        num_update=self.num_update,
                        num_epoch_list=num_epoch_list
                    )
                    batch_end_callback(batch_end_params)

                nbatch += 1

    def get_params(self):
        _g, _x = self.backbone_module.get_params()
        g = _g.copy()
        x = _x.copy()
        # _g, _x = self.bn_module.get_params()
        # ag = _g.copy()
        # ax = _x.copy()
        # g.update(ag)
        # x.update(ax)
        return g, x

    def get_export_params(self):
        assert self.binded and self.params_initialized
        _g, _x = self.backbone_module.get_params()
        g = _g.copy()
        x = _x.copy()
        # _g, _x = self.bn_module.get_params()
        # ag = _g.copy()
        # ax = _x.copy()
        # g.update(ag)
        # x.update(ax)
        return g, x

    def get_ndarray2(self, context, name, arr):
        key = "%s_%s" % (name, context)
        if key not in self.nd_cache:
            v = nd.zeros(shape=arr.shape, ctx=context, dtype=arr.dtype)
            self.nd_cache[key] = v
        else:
            v = self.nd_cache[key]
        arr.copyto(v)
        return v

    def get_ndarray(self, context, name, shape, dtype='float32'):
        key = "%s_%s" % (name, context)
        if key not in self.nd_cache:
            v = nd.zeros(shape=shape, ctx=context, dtype=dtype)
            self.nd_cache[key] = v
        else:
            v = self.nd_cache[key]
        return v

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        assert self.binded
        # backbone
        self.backbone_module.init_params(
            initializer=initializer, arg_params=arg_params,
            aux_params=aux_params, allow_missing=allow_missing,
            force_init=force_init, allow_extra=allow_extra)

        self.backbone_module.init_params(
            initializer=initializer, arg_params=None,
            aux_params=None, allow_missing=allow_missing,
            force_init=force_init, allow_extra=allow_extra)

        # self.bn_module.init_params(
        #     initializer=initializer, arg_params=arg_params,
        #     aux_params=aux_params, allow_missing=allow_missing,
        #     force_init=force_init, allow_extra=allow_extra)
        self.params_initialized = True

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True,
                   allow_extra=False):
        self.init_params(
            initializer=None, arg_params=arg_params, aux_params=aux_params,
            allow_missing=allow_missing, force_init=force_init,
            allow_extra=allow_extra)

    def save_params(self, fname):
        arg_params, aux_params = self.get_params()
        save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
        ndarray.save(fname, save_dict)

    def load_params(self, fname):
        save_dict = ndarray.load(fname)
        arg_params = {}
        aux_params = {}
        for k, value in save_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                arg_params[name] = value
            elif arg_type == 'aux':
                aux_params[name] = value
            else:
                raise ValueError("Invalid param file " + fname)
        self.set_params(arg_params, aux_params)

    def get_states(self, merge_multi_context=True):
        raise NotImplementedError

    def set_states(self, states=None, value=None):
        raise NotImplementedError

    def prepare(self, data_batch, sparse_row_id_fn=None):
        if sparse_row_id_fn is not None:
            warnings.warn(UserWarning("sparse_row_id_fn is not invoked for BaseModule."))

    def allgather(self, tensor, name, shape, dtype, context):
        assert isinstance(tensor, nd.NDArray),          type(tensor)
        assert isinstance(name, str),                   type(name)
        assert isinstance(shape, tuple),                type(shape)
        assert isinstance(dtype, str),                  type(dtype)
        assert isinstance(context, mx.context.Context), type(context)
        """
        Implement in-place AllGather using AllReduce
        """
        total_tensor = self.get_ndarray(
            context=context, name=name, shape=shape, dtype=dtype)
        total_tensor[:] = 0                                      # reset array before all-reduce is very important
        total_tensor[self.rank * self.batch_size:
                     self.rank * self.batch_size + self.batch_size] = tensor
        hvd.allreduce_(total_tensor, average=False)              # all-reduce in-place
        return total_tensor

    # pylint: enable=unused-argument
    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized
        self.backbone_module.forward(data_batch, is_train=is_train)
        if is_train:
            self.num_update += 1
            fc1 = self.backbone_module.get_outputs()[0]
            label = data_batch.label[0]

            total_features = self.allgather(
                tensor=fc1,
                name='total_feature',
                shape=(self.batch_size * self.size, self.embedding_size),
                dtype='float32',
                context=self.gpu
            )
            total_labels = self.allgather(
                tensor=label,
                name='total_label',
                shape=(self.batch_size * self.size,),
                dtype='int32',
                context=self.cpu
            )

            # self.bn_module.forward(mx.io.DataBatch([total_features],  []), is_train=True)
            # total_features = self.bn_module.get_outputs(merge_multi_context=True)[0]
            return total_features, total_labels
        else:
            return None
            # raise ValueError

    def backward_all(self, total_feature, total_label, ):
        # get memory bank learning rate
        self.memory_lr = self.memory_optimizer.lr_scheduler(self.num_update)

        # reverse shuffle bn
        total_feature = total_feature.reshape(-1, self.embedding_size * self.head_num)
        # global_label
        total_label = total_label.reshape(-1, self.head_num)
        #
        self.grad_cache = self.get_ndarray(self.gpu, 'grad_cache', total_feature.shape)
        self.loss_cache = self.get_ndarray(self.gpu, 'loss_cache', [self.head_num])

        self.grad_cache[:] = 0
        self.loss_cache[:] = 0

        for head_id in range(self.head_num):
            _fc1_one_head = total_feature[
                            :,
                            head_id * self.embedding_size:
                            head_id * self.embedding_size + self.embedding_size
                            ]
            _label_one_head = total_label[:, head_id]

            grad, loss = self.backward(head_id, _fc1_one_head, _label_one_head)
            self.grad_cache[
                :,
                head_id * self.embedding_size:
                head_id * self.embedding_size + self.embedding_size
            ] = grad
            self.loss_cache[head_id] = loss

        total_feature_grad = self.grad_cache.reshape(-1, self.embedding_size)
        total_feature_grad = hvd.allreduce(total_feature_grad, average=False)

        # self.bn_module.backward(out_grads=[total_feature_grad / self.backbone_grad_rescale])
        # bn_input_grad = self.bn_module.get_input_grads()[0]

        fc1_grad = total_feature_grad[
            self.batch_size * self.rank:
            self.batch_size * self.rank + self.batch_size
        ]
        self.backbone_module.backward(out_grads=[fc1_grad])

    def backward(self, head_id, fc1, label):

        memory_bank = self.memory_bank_list[head_id]
        this_rank_classes = int(memory_bank.num_sample)
        local_index, unique_sorted_global_label = memory_bank.sample(label)

        # Get local index
        _mapping_dict = {}
        local_sampled_class = local_index + self.rank * memory_bank.num_local
        global_label_set = set(unique_sorted_global_label)
        for idx, absolute_label in enumerate(local_sampled_class):
            if absolute_label in global_label_set:
                _mapping_dict[absolute_label] = idx + self.rank * memory_bank.num_sample

        label_list = list(label.asnumpy())
        mapping_label = []
        for i in range(len(label_list)):
            absolute_label = label_list[i]
            if absolute_label in _mapping_dict.keys():
                mapping_label.append(_mapping_dict[absolute_label])
            else:
                mapping_label.append(-1)

        mapping_label = nd.array(mapping_label, dtype=np.int32)

        # Get weight
        local_index = nd.array(local_index)
        local_index = self.get_ndarray2(self.gpu, "local_index_%d" % head_id, local_index)
        sample_weight, sample_weight_mom = memory_bank.get(local_index)

        # Sync to gpu
        if memory_bank.gpu:
            _data       = self.get_ndarray2(self.gpu, "data_%d_%d"       % (self.rank, head_id), fc1)
            _weight     = self.get_ndarray2(self.gpu, 'weight_%d_%d'     % (self.rank, head_id), sample_weight)
            _weight_mom = self.get_ndarray2(self.gpu, 'weight_mom_%d_%d' % (self.rank, head_id), sample_weight_mom)
        else:
            _data       = self.get_ndarray2(self.gpu, "data_%d_%d"       % (self.rank, head_id), fc1)
            _weight     = self.get_ndarray2(self.gpu, 'weight_%d_%d'     % (self.rank, head_id), sample_weight)
            _weight_mom = self.get_ndarray2(self.gpu, 'weight_mom_%d_%d' % (self.rank, head_id), sample_weight_mom)

        # Attach grad
        _data.attach_grad()
        _weight.attach_grad()

        # Convert label
        _label = self.get_ndarray2(self.gpu, 'mapping_label_%d_%d' % (self.rank, head_id), mapping_label)
        _label = _label - int(self.rank * memory_bank.num_sample)
        _fc7, _one_hot = self.fc7_model.forward(_data, _weight, mapping_label=_label, depth=this_rank_classes)

        # Sync max
        max_fc7 = nd.max(_fc7, axis=1, keepdims=True)
        max_fc7 = nd.reshape(max_fc7, -1)

        total_max_fc7 = self.get_ndarray(
            context=self.gpu, name='total_max_fc7_%d' % head_id,
            shape=(max_fc7.shape[0], self.size), dtype='float32')
        total_max_fc7[:] = 0
        total_max_fc7[:, self.rank] = max_fc7
        hvd.allreduce_(total_max_fc7, average=False)

        global_max_fc7 = self.get_ndarray(
            context=self.gpu, name='global_max_fc7_%d' % head_id,
            shape=(max_fc7.shape[0], 1), dtype='float32')
        nd.max(total_max_fc7, axis=1, keepdims=True, out=global_max_fc7)

        # Calculate prob
        _fc7_grad = nd.broadcast_sub(_fc7, global_max_fc7)
        _fc7_grad = nd.exp(_fc7_grad)

        # Calculate sum
        sum_fc7 = nd.sum(_fc7_grad, axis=1, keepdims=True)
        global_sum_fc7 = hvd.allreduce(sum_fc7, average=False)

        # Calculate grad
        _fc7_grad = nd.broadcast_div(_fc7_grad, global_sum_fc7)

        # Calculate loss
        tmp = _fc7_grad * _one_hot
        tmp = nd.sum(tmp, axis=1, keepdims=True)
        tmp = self.get_ndarray2(self.gpu, 'ctx_loss_%d' % head_id, tmp)
        tmp = hvd.allreduce(tmp, average=False)
        global_loss = -nd.mean(nd.log(tmp + 1e-30))

        _fc7_grad = _fc7_grad - _one_hot

        # Backward
        _fc7.backward(out_grad=_fc7_grad)

        # Update center
        _weight_grad = _weight.grad
        self.memory_optimizer.update(weight=_weight, grad=_weight_grad, state=_weight_mom,
                                     learning_rate=self.memory_lr * self.memory_lr_scale_list[head_id])
        if memory_bank.gpu:
            memory_bank.set(
                index=local_index,
                updated_weight=_weight,
                updated_weight_mom=_weight_mom)
        else:
            memory_bank.set(
                index=local_index,
                updated_weight     = self.get_ndarray2(mx.cpu(), "cpu_weight_%d_%d"     % (self.rank, head_id), _weight),
                updated_weight_mom = self.get_ndarray2(mx.cpu(), "cpu_weight_mom_%d_%d" % (self.rank, head_id), _weight_mom))
        return _data.grad, global_loss

    def get_outputs(self, merge_multi_context=True):
        return self.backbone_module.get_outputs(merge_multi_context=merge_multi_context)

    def update(self):
        self.backbone_module.update()
        # self.bn_module.update()
        mx.nd.waitall()

    def bind(self, data_shapes_list=None, label_shapes_list=None, for_training=True,
             inputs_need_grad=False):
        assert data_shapes_list is not None and label_shapes_list is not None
        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return
        data_name = data_shapes_list[0][0][0]
        data_shapes = data_shapes_list[0][0][1]
        label_name = label_shapes_list[0][0][0]
        label_shapes = label_shapes_list[0][0][1]

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        _backbone_data_shapes = [(data_name, (self.batch_size,) + data_shapes[1:])]
        _backbone_label_shapes = [(label_name, (self.batch_size,) + label_shapes[1:])]

        _bn_data_shapes = [(data_name, (self.batch_size * self.size, self.embedding_size))]
        self.backbone_module.bind(
            data_shapes=_backbone_data_shapes,
            label_shapes=_backbone_label_shapes,
            for_training=for_training,
            inputs_need_grad=inputs_need_grad)
        # self.bn_module.bind(
        #     data_shapes=_bn_data_shapes,
        #     for_training=for_training,
        #     inputs_need_grad=True
        # )

    def init_optimizer(self, optimizer_params, force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return
        # backbone
        # optimizer_backbone = DistributedOptimizer(LARS(**optimizer_params))
        # optimizer_bn       = DistributedOptimizer(LARS(**optimizer_params), prefix='bn_')

        optimizer_backbone = DistributedOptimizer(SGD(**optimizer_params))
        self.backbone_module.init_optimizer(
            'local', optimizer_backbone, force_init=force_init)
        # optimizer_bn = DistributedOptimizer(SGD(**optimizer_params), prefix='bn_')
        # self.bn_module.init_optimizer(
        #     'local', optimizer_bn,       force_init=force_init)
        self.optimizer_initialized = True

