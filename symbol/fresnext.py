# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu
Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network"
'''
from default import config
import mxnet as mx
import numpy as np


def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    elif act_type == 'leaky':
        body = mx.sym.LeakyReLU(data=data, act_type='leaky', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256,
                  memonger=False, attr={}):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    act_type = 'relu'
    # act_type = 'leaky'
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper

        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1),
            pad=(0, 0),
            # conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter), kernel=(1,1), stride=(1,1), pad=(0,0),
            no_bias=True, workspace=workspace, name=name + '_conv1', attr=attr)
        if memonger:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1',
                cudnn_off=True)
        else:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=num_group, kernel=(3, 3),
            stride=stride, pad=(1, 1),
            # conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1),
            no_bias=True, workspace=workspace, name=name + '_conv2', attr=attr)
        if memonger:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2',
                cudnn_off=True)
        else:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        # act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')

        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
            no_bias=True,
            workspace=workspace, name=name + '_conv3', attr=attr)
        if memonger:
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3',
                cudnn_off=True)
        else:
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                no_bias=True,
                workspace=workspace, name=name + '_sc', attr=attr)
            if memonger:
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                    name=name + '_sc_bn', cudnn_off=True)
            else:
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                    name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn3 + shortcut
        # return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
        return Act(data=eltwise, act_type=act_type, name=name + '_relu')
    else:

        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
            no_bias=True, workspace=workspace, name=name + '_conv1', attr=attr)
        if memonger:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1',
                cudnn_off=True)
        else:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
            no_bias=True, workspace=workspace, name=name + '_conv2', attr=attr)
        if memonger:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2',
                cudnn_off=True)
        else:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                no_bias=True,
                workspace=workspace, name=name + '_sc', attr=attr)
            if memonger:
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                    name=name + '_sc_bn', cudnn_off=True)
            else:
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                    name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        # return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
        return Act(data=eltwise, act_type=act_type, name=name + '_relu')


def resnext(units, num_stages, filter_list, num_classes, num_group, bottle_neck=True, bn_mom=0.9, workspace=256,
            dtype='float32', memonger=False, attr={}):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    num_groupes: int
    Number of conv groups
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """
    act_type = 'relu'
    num_unit = len(units)
    assert (num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = data - 127.5
    data = data * 0.0078125

    if config.fp16:
        data = mx.sym.Cast(data=data, dtype=np.float16)

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        no_bias=True, name="conv0", workspace=workspace, attr=attr)
    if memonger:
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', cudnn_off=True)
    else:
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (2, 2), False,
            name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
            bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

    if config.fp16:
        body = mx.sym.Cast(data=body, dtype=np.float32)

    body = mx.sym.Convolution(data=body, num_filter=int(num_classes / 8), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
        no_bias=True, name="conv_final", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')

    return fc1


def get_symbol(num_layers, embedding_size, num_group=32, conv_workspace=256, dtype='float32', memonger=False, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    num_classes = embedding_size
    num_layers =num_layers
    if 1:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnext(
        units=units,
        num_stages=num_stages,
        filter_list=filter_list,
        num_classes=num_classes,
        num_group=num_group,
        bottle_neck=bottle_neck,
        workspace=conv_workspace,
        dtype=dtype,
        memonger=memonger)