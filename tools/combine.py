from mxnet import nd
import os


def _load_params(fname):
    array = nd.load(fname)[0]
    return array


def _cat(root, suffix):
    res = []
    for rank in range(8):
        array = _load_params(os.path.join(root, "%d_%s" % (rank, suffix)))
        res.append(array)
    concat_array = nd.concat(*res, dim=0)
    print("%s name shape is" % suffix, concat_array.shape)
    nd.save(os.path.join(root, "%s_%s" % ("concat", suffix)), concat_array)


if __name__ == '__main__':
    #
    root = "/anxiang/workspace/class_center_5_15"
    #
    [_cat(root, fname) for fname in
     [
         "celeb_largeFC.param",
         "faces_msw_largeFC.param"
     ]
    ]