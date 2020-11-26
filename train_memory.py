# deepglint
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys
import os

import horovod.mxnet as hvd
import mxnet as mx
import math

import default
from image_iter import FaceImageIter, get_iter
from memory_ModuleGlint import MemoryModuleGlint
from callbacks import ModelSaveCallBack, LogCallBack, CenterSaveCallBack, VertificationCallBack
from default import config
from memory_bank import MemoryBank
from memory_optimizer import MemoryBankLarsOptimizer, MemoryBankSGDOptimizer
from memory_scheduler import get_scheduler
from utils.set_logger import set_logger
from memory_softmax import PartialSoftMax

sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train parall face network')
    # general
    parser.add_argument('--dataset', default='xj20w', help='dataset config')
    parser.add_argument('--network', default='r50', help='network config')
    parser.add_argument('--loss', default='cosface', help='loss config')

    args, rest = parser.parse_known_args()
    default.generate_config(args.loss, args.dataset, args.network)
    parser.add_argument('--models-root', default="./test", help='root directory to save model.')
    #
    parser.add_argument('--partition', type=int, default=8000000, help='byte scheduler')
    parser.add_argument('--credit', type=float, default=16000000, help='byte scheduler')
    args = parser.parse_args()
    return args


def get_symbol_bn():
    embedding = mx.symbol.Variable('data')
    embedding = mx.sym.BatchNorm(data=embedding, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    return embedding


def get_symbol_embedding():
    from symbol import fresnet, resnet
    embedding = eval(config.net_name).get_symbol()
    all_label = mx.symbol.Variable('softmax_label')
    all_label = mx.symbol.BlockGrad(all_label)
    out_list = [embedding, all_label]
    out = mx.symbol.Group(out_list)
    save_symbol = mx.sym.BatchNorm(data=embedding, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    return out, save_symbol


def train_net(args):
    # Horovod: initialize Horovod
    hvd.init()

    # local rank & rank
    local_rank = hvd.local_rank()
    rank = hvd.rank()
    size = hvd.size()

    prefix = os.path.join(args.models_root, 'model')
    prefix_dir = os.path.dirname(prefix)

    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    set_logger(logger, rank, prefix_dir)

    head_num = len(config.head_name_list)
    assert config.batch_size % head_num == 0

    data_shape = tuple(config.data_shape)
    batch_size = config.batch_size
    embedding_size = config.embedding_size

    head_batch_size = int(batch_size // head_num)
    memory_bank_list = []
    train_data_iter_list = []
    for head_id in range(head_num):
        # Divide the class centers equally among each workers
        num_local = (config.num_classes_list[head_id] + size - 1) // size
        # num_local = int(num_local)
        num_local = math.floor(num_local)

        # Number of class centers sampled per worker must be an integer multiple of gpu num
        # num_sample = int(num_local * config.sample_ratio // 8 * 8)
        num_sample = int(num_local * config.sample_ratio)
        _memory_bank = MemoryBank(
            num_sample=num_sample,
            num_local=num_local,
            rank=rank,
            local_rank=local_rank,
            name=config.head_name_list[head_id],
            embedding_size=embedding_size,
            prefix=prefix_dir)

        memory_bank_list.append(_memory_bank)
        rec_path = config.rec_list[head_id]
        idx_path = config.rec_list[head_id][0:-4] + ".idx"
        # train_iter = FaceImageIter(
        #     batch_size=head_batch_size,
        #     data_shape=data_shape,
        #     path_imgrec=rec_path,
        #     shuffle=True,
        #     rand_mirror=True,
        #     context=rank,
        #     context_num=hvd.size()
        # )
        train_iter = get_iter(rec_path, idx_path, data_shape, head_batch_size, hvd.size(), rank, local_rank)
        train_data_iter = mx.io.PrefetchingIter(train_iter)
        train_data_iter_list.append(train_data_iter)

    esym, save_symbol = get_symbol_embedding()
    margins = (config.loss_m1, config.loss_m2, config.loss_m3)
    fc7_model = PartialSoftMax(margins, config.loss_s, embedding_size)

    # optimizer
    # backbone  lr_scheduler & optimizer
    backbone_lr_scheduler, memory_bank_lr_scheduler = get_scheduler()

    backbone_grad_rescale = size
    backbone_kwargs = {
        'learning_rate': config.backbone_lr,
        'momentum': 0.9,
        'wd': 5e-4,
        'rescale_grad': 1.0 / (config.batch_size * size) * backbone_grad_rescale,
        'multi_precision': config.fp16,
        'lr_scheduler': backbone_lr_scheduler,
    }

    # memory_bank lr_scheduler & optimizer
    memory_bank_optimizer = MemoryBankSGDOptimizer(
        lr_scheduler=memory_bank_lr_scheduler,
        rescale_grad=1.0 / (config.batch_size / head_num) / hvd.size(),
    )

    # model
    model = MemoryModuleGlint(
        symbol=esym,
        bn_symbol=get_symbol_bn(),
        batch_size=batch_size,
        size=size,
        rank=rank,
        local_rank=local_rank,
        fc7_model=fc7_model,  # FIXME
        memory_bank_list=memory_bank_list,
        memory_optimizer=memory_bank_optimizer,
        embedding_size=embedding_size,
        head_num=head_num,
        backbone_grad_rescale=backbone_grad_rescale,
        memory_lr_scale_list=config.memory_lr_scale_list
    )

    if os.environ.get('USE_BYTESCHEDULER') is not None and os.environ.get('USE_BYTESCHEDULER') == "1":
        if args.partition:
            os.environ["BYTESCHEDULER_PARTITION"] = str(args.partition)
        if args.credit:
            os.environ["BYTESCHEDULER_CREDIT"] = str(args.credit)
        import bytescheduler.mxnet.horovod as bsc
        bsc.init()

    # resnet style
    initializer = mx.init.Normal(0.1)

    #
    if config.verbose_flag:
        cb_vert = VertificationCallBack(esym, config.verbose, model, config.val_targets, config.rec_list[0][0:-10],
                                    (112, 112))
    cb_speed = LogCallBack(batch_size, config.head_name_list, rank, size, prefix_dir, config.frequent)
    cb_save = ModelSaveCallBack(save_symbol, model, prefix, config.max_update, rank)
    cb_center_save = CenterSaveCallBack(memory_bank_list)

    if config.verbose_flag:
        def call_back(params):
            cb_speed(params)
            cb_vert(params)
            cb_center_save(params)
            cb_save(params)
    else:
        def call_back(params):
            cb_speed(params)
            cb_center_save(params)
            cb_save(params)

    model.fit(
        train_data_iter_list,
        begin_epoch=0,
        num_epoch=999999,
        optimizer_params=backbone_kwargs,
        initializer=initializer,
        allow_missing=True,
        batch_end_callback=call_back,
    )


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    os.environ['MXNET_BACKWARD_DO_MIRROR'] = '0'
    main()
