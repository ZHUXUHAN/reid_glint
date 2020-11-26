import mxnet as mx


class MetricNdarray(object):
    def __init__(self):
        self.sum = None
        self.count = 0
        self.reset()

    def reset(self):
        self.sum = None
        self.count = 0

    def update(self, val, n=1):
        assert isinstance(val, mx.nd.NDArray), type(val)
        if self.sum is None:                             # init sum
            self.sum = mx.nd.zeros_like(val)

        self.sum += val * n
        self.count += n

    def get(self):
        average = self.sum / self.count
        return average.asscalar()