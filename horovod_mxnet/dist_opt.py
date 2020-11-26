import horovod.mxnet as hvd
import mxnet as mx


# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer, prefix=""):
        self._optimizer = optimizer
        self._prefix = prefix

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_allreduce(self, index, grad):
        if hvd.size() == 1:
            return

        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                hvd.allreduce_(grad[i], average=False,
                    name=self._prefix + str(index[i]), priority=-i)
        else:
            hvd.allreduce_(grad, average=False, name=self._prefix + str(index))

    def update(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)
