import mxnet as mx
from mxnet import nd


def test_index(array, idx, i):
    sim = nd.dot(array[i], array, transpose_b=True)
    index = idx[i]
    index = index[:30]
    print('===============================')
    print(sim[index])
    sim = nd.sort(sim, is_ascend=0)
    print("mean all", nd.mean(sim))
    print("mean 100", nd.mean(sim[:100]))
    print("mean 500", nd.mean(sim[:500]))
    print("mean 1000", nd.mean(sim[:1000]))
    print("mean 10000", nd.mean(sim[:10000]))
    print("mean 20000", nd.mean(sim[:20000]))
    print("mean -1000", nd.mean(sim[len(sim) - 1000:]))

    # print(sim.shape)


if __name__ == '__main__':
    prefix = "/anxiang/workspace/class_center_5_15/concat_celeb_largeFC"
    array = nd.load("%s.param" % prefix)[0]
    array = nd.L2Normalization(array)
    idx = nd.load("%s.index" % prefix)[0]
    for i in [12, 32, 43, 23, 123, 554, 41213, 45, 657, 234, 54, 23]:
        test_index(array, idx, i)
