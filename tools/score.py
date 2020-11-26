#! /home/xiangan/miniconda3/envs/mxnet/bin/python
import mxnet as mx
from mxnet import nd

BATCH_SIZE = 128
CONTEXT_NUM = 8  # gpu number


def load(fname):
    _arr = nd.load(fname)[0]
    _arr = nd.L2Normalization(_arr)
    return _arr


def score(x, y):
    _sim = nd.dot(x, y, transpose_b=True)
    _indices = nd.argsort(-_sim, axis=1)
    return _sim, _indices[:, 1: 100]


def get(prefix):
    array: nd.NDArray = load("%s.param" % prefix)
    arrays_on_gpu = [array.as_in_context(mx.gpu(i)) for i in range(CONTEXT_NUM)]
    num_classes = array.shape[0]
    idx = 0
    result = []
    while idx < num_classes:
        for i in range(CONTEXT_NUM):
            try:
                sim, indices = score(arrays_on_gpu[i][idx:idx + BATCH_SIZE], arrays_on_gpu[i])
                result.append(indices.as_in_context(mx.cpu(i)))
                idx += BATCH_SIZE
            except IndexError:
                sim, indices = score(arrays_on_gpu[i][idx:], arrays_on_gpu[i])
                result.append(indices.as_in_context(mx.cpu(i)))
                idx += BATCH_SIZE
                break

    index_matrix = nd.concat(*result, dim=0)
    nd.save("%s.index" % prefix, index_matrix)


if __name__ == '__main__':
    get("/anxiang/workspace/class_center_5_15/concat_celeb_largeFC")
    get("/anxiang/workspace/class_center_5_15/concat_faces_msw_largeFC")