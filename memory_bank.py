import os

import numpy as np
from mxnet import nd
import mxnet as mx

from memory_samplers import WeightIndexSampler
from default_hard_config import hard_config


class MemoryBank(object):
    """ Memory class center
    """
    def __repr__(self):
        return ""

    def __init__(self, num_sample, num_local, rank, local_rank, name, embedding_size, prefix, gpu=True):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.name = name
        self.embedding_size = embedding_size
        self.gpu = gpu
        self.prefix = prefix
        if gpu:
            self.weight = nd.random_normal(
                loc=0, scale=0.01, shape=(self.num_local, self.embedding_size),
                ctx=mx.gpu(local_rank)
            )
            self.weight_mom = nd.zeros_like(self.weight)
        else:
            self.weight = nd.random_normal(
                loc=0, scale=0.01, shape=(self.num_local, self.embedding_size))
            self.weight_mom = nd.zeros_like(self.weight)
        self.weight_index_sampler = WeightIndexSampler(num_sample, num_local, rank, name)
        pass

    def sample(self, global_label):
        assert isinstance(global_label, nd.NDArray)
        global_label = global_label.asnumpy()
        global_label = np.unique(global_label)
        global_label.sort()
        index = self.weight_index_sampler(global_label)
        index.sort()
        return index, global_label

    def get(self, index):
        return self.weight[index], self.weight_mom[index]

    def set(self, index, updated_weight, updated_weight_mom=None):
        self.weight[index] = updated_weight
        self.weight_mom[index] = updated_weight_mom

    def save(self):
        # if hard server
        if hard_config.HARD_SERVER:
            # we just need weight to find those hard classes
            nd.save(
                fname=os.path.join(hard_config.PREFIX, "%d_%s_largeFC.param" % (self.rank, self.name)),
                data=self.weight
            )
        else:
            nd.save(
                fname=os.path.join(self.prefix, "%d_%s_largeFC.param" % (self.rank, self.name)),
                data=self.weight
            )
        nd.save(
            fname=os.path.join(self.prefix, "%d_%s_largeFC_mom.param" % (self.rank, self.name)),
            data=self.weight_mom
        )

    def load(self):
        raise NotImplementedError
