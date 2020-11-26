import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from mxnet import nd
from tqdm import tqdm
import sys
sys.path.append('../')
from default_hard_config import hard_config

HEAD_NAME_LIST = hard_config.HEAD_NAME_LIST
BATCH_SIZE = 400
FILE_NUM = 64
GPU_NUM = 8  # gpu number
device = [torch.device(i % GPU_NUM) for i in range(FILE_NUM)]
torch.backends.cudnn.benchmark = True


def score(x, arrays_on_gpu):
    device_of_x = x.device
    x = [x.to(device[i]) for i in range(FILE_NUM)]
    all_value = [0] * FILE_NUM
    all_index = [0] * FILE_NUM
    for i in range(FILE_NUM):
        bs_score = torch.einsum("ik, jk -> ij", x[i], arrays_on_gpu[i])
        all_value[i], all_index[i] = torch.topk(bs_score, k=hard_config.TOP_K, dim=1)
        all_index[i].add_(arrays_on_gpu[i].shape[0] * i)

    for i in range(FILE_NUM):
        all_value[i] = all_value[i].to(device_of_x)
        all_index[i] = all_index[i].to(device_of_x)

    last_value = torch.cat(all_value, dim=1)
    last_index = torch.cat(all_index, dim=1)
    last_value, index = last_value.topk(dim=1, k=hard_config.TOP_K)
    last_index = torch.gather(last_index, dim=1, index=index)

    return last_index


@torch.no_grad()
def get(root, suffix):
    arrays_on_gpu = [0] * FILE_NUM
    for i in range(FILE_NUM):
        file_i = os.path.join(root, str(i) + suffix)
        arrays_on_gpu[i] = torch.Tensor(nd.load(file_i)[0].asnumpy()).to(device[i])

    for i in range(FILE_NUM):
        F.normalize(arrays_on_gpu[i], p=2, dim=1, out=arrays_on_gpu[i])

    print('finish load')
    print(sum([arr.shape[0] for arr in arrays_on_gpu]))
    pbar = tqdm(total=sum([arr.shape[0] for arr in arrays_on_gpu]))
    similar_matrixs = []
    for i in range(FILE_NUM):
        result = []
        num_classes = arrays_on_gpu[i].size()[0]
        idx = 0
        while idx < num_classes:
            indices = score(arrays_on_gpu[i][idx:min(idx + BATCH_SIZE, num_classes)], arrays_on_gpu)
            result.append(indices)
            idx += BATCH_SIZE
            pbar.update(BATCH_SIZE)
        similar_matrixs.append(torch.cat(result, dim=0))
    pbar.close()

    result = []
    for i in range(len(similar_matrixs)):
        result.append(similar_matrixs[i][:, 1:].cpu().numpy())
    result = np.concatenate(result, axis=0)
    result = result.astype(np.int32)

    # np.save('/anxiang/tmp/xj600w_largeFC.param.npy', result)
    np.save(os.path.join(hard_config.PREFIX, '%s_largeFC.param.npy' % name), result)


def is_modify(root, last_time_dict, name):
    for i in range(FILE_NUM):
        _file_name = os.path.join(root, '%d_%s_largeFC.param' % (i, name))
        if not os.path.exists(_file_name):
            return False
        if os.path.getmtime(_file_name) == last_time_dict[name][i]:
            return False
    for i in range(FILE_NUM):
        _file_name = os.path.join(root, '%d_%s_largeFC.param' % (i, name))
        last_time_dict[name][i] = os.path.getmtime(_file_name)
    return True


if __name__ == '__main__':
    #
    last_time_dict = {}
    for name in HEAD_NAME_LIST:
        last_time_dict[name] = [0] * FILE_NUM
    #
    count = 1
    while True:
        for name in HEAD_NAME_LIST:
            print('try compute similar_matrixs ', count)
            count += 1
            if is_modify(hard_config.PREFIX, last_time_dict, name):
                try:
                    get(root=hard_config.PREFIX, suffix='_%s_largeFC.param' % name)
                except BaseException as arg:
                    print(arg)
                    for i in range(FILE_NUM):
                        last_time_dict[name][i] = -1
        time.sleep(600)


