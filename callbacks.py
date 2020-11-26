import logging
import os
import sys
import time

import mxnet as mx
from mxboard import SummaryWriter
from mxnet import nd


from eval import verification
from default_hard_config import hard_config
from utils.metric_utils import MetricNdarray


class VertificationCallBack(object):
    def __init__(self, symbol, verbose, model, val_targets, data_dir, image_size):
        self.verbose = verbose
        self.symbol = symbol
        self.highest_acc = 0.0
        self.highest_acc_list = [0.0] * len(val_targets)
        self.model = model
        self.ver_list = []
        self.ver_name_list = []
        self.init_dataset(val_targets=val_targets, data_dir=data_dir, image_size=image_size)

    def ver_test(self, num_update):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], self.model, 10, 10, None, None)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], num_update, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], num_update, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info('[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], num_update, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, param):
        #
        num_update = param.num_update
        #
        if num_update > 0 and num_update % self.verbose == 0:  # debug in mbatches in 100 and 200
            # accuracy list
            self.ver_test(num_update)


class CenterSaveCallBack(object):
    def __init__(self, memory_bank_list, save_interval=10000):
        self.save_interval = save_interval
        self.memory_bank_list = memory_bank_list

    def __call__(self, param):
        if hard_config.HARD_SERVER:
            if param.num_update % hard_config.SAVE_INTERVAL == 0:
                for _memory_bank in self.memory_bank_list:
                    _memory_bank.save()
        else:
            if param.num_update % self.save_interval == 0:
                for _memory_bank in self.memory_bank_list:
                    _memory_bank.save()


class ModelSaveCallBack(object):
    def __init__(self, symbol, model, prefix, max_step, rank):
        self.symbol = symbol
        self.model = model
        self.prefix = prefix
        self.max_step = max_step
        self.rank = rank

    def __call__(self, param):
        num_update = param.num_update

        if num_update in [self.max_step - 10, ] or (num_update % 10000 == 0 and num_update > 0):
            if self.rank == 0:
                # params
                arg, aux = self.model.get_export_params()
                # symbol
                _sym = self.symbol
                # save
                mx.model.save_checkpoint(
                    prefix=self.prefix, epoch=0, symbol=_sym,
                    arg_params=arg, aux_params=aux)

        # training is over
        if num_update > self.max_step > 0:
            logging.info('Training is over!')
            sys.exit(0)


class LogCallBack(object):
    def __init__(self, batch_size, head_name_list, rank, size, prefix_dir, frequent):
        self.batch_size = batch_size
        self.rank = rank
        self.size = size
        self.prefix_dir = prefix_dir
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        #
        self.head_name_list = head_name_list
        self.loss_metric_list = [MetricNdarray() for x in head_name_list]
        t = time.localtime()

        self.summary_writer = SummaryWriter(
            logdir=os.path.join(self.prefix_dir, 'log_tensorboard', str(t.tm_mon)+'_'+str(t.tm_mday) \
                                +'_'+str(t.tm_hour)),
            verbose=False)

    def __call__(self, param):
        self.logging(param)

    def logging(self, param):
        """Callback to Show speed."""
        count = param.num_update

        if self.last_count > count:
            self.init = False
        self.last_count = count

        loss_list = param.loss_list
        for i in range(len(self.head_name_list)):
            self.loss_metric_list[i].update(loss_list[i])

        if self.init:
            if count % self.frequent == 0:
                nd.waitall()
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.size
                except ZeroDivisionError:
                    speed = float('inf')
                    speed_total = float('inf')

                loss_str_format = ""
                #
                for idx, name in enumerate(self.head_name_list):
                    loss_scalar = self.loss_metric_list[idx].get()

                    # summary loss
                    self.summary_writer.add_scalar(
                        tag="%s_loss" % name,
                        value=loss_scalar, global_step=param.num_update)
                    _ = "[%d][%s]:%.2f " % (param.num_epoch_list[idx], name, loss_scalar)
                    loss_str_format += _
                    self.loss_metric_list[idx].reset()
                # summary speed
                self.summary_writer.add_scalar(
                    tag="speed",
                    value=speed, global_step=param.num_update)
                self.summary_writer.flush()
                if self.rank == 0:
                    logging.info(
                        "Iter:%d Rank:%.2f it/sec Total:%.2f it/sec %s",
                        param.num_update, speed, speed_total, loss_str_format)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()