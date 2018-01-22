#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pdb
import cv2
import sys
import argparse
import numpy as np
import os
import shutil
import multiprocessing
import json

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

try:
    from .cfgs.config import cfg
    from .evaluate import do_python_eval
    from .yolo_utils import YoloModel, get_data, get_config
except Exception:
    from cfgs.config import cfg
    from evaluate import do_python_eval
    from yolo_utils import YoloModel, get_data, get_config

@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, nl=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return nl(conv, name='output')

@under_name_scope()
def channel_shuffle(l, group):
    if args.multi_scale:
        in_shape = tf.shape(l)
        in_channel = l.get_shape().as_list()[1]
        in_h = in_shape[2]
        in_w = in_shape[3]
        l = tf.reshape(l, (-1, group, in_channel // group, in_h, in_w))
        l = tf.transpose(l, [0, 2, 1, 3, 4])
        l = tf.reshape(l, (-1, in_channel, in_h, in_w))
    else:
        in_shape = l.get_shape().as_list()
        in_channel = in_shape[1]
        l = tf.reshape(l, [-1, group, in_channel // group] + in_shape[-2:])
        l = tf.transpose(l, [0, 2, 1, 3, 4])
        l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
    return l

def BN(x, name):
    return BatchNorm('bn', x)

class ShufflenetYolo(YoloModel):

    def get_logits(self, image):

        def shufflenet_unit(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 16 else 1
            l = Conv2D('conv1', l, out_channel // 4, 1, split=first_split, nl=BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, 3, nl=BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel if stride == 1 else out_channel - in_channel,
                       1, split=group, nl=BN)
            if stride == 1:     # unit (b)
                output = tf.nn.relu(shortcut + l)
            else:   # unit (c)
                shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
                output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
            return output

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope(Conv2D, use_bias=False):
            group = 8
            channels = [384, 768, 1536]
            # channels = [224, 416, 832]

            l = Conv2D('conv1', image, 16, 3, stride=2, nl=BNReLU)
            l = MaxPooling('pool1', l, 3, 2, padding='SAME')

            with tf.variable_scope('group1'):
                for i in range(4):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[0], group, 2 if i == 0 else 1)

            with tf.variable_scope('group2'):
                for i in range(8):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[1], group, 2 if i == 0 else 1)

            high_res = l

            with tf.variable_scope('group3'):
                for i in range(4):
                    with tf.variable_scope('block{}'.format(i)):
                        l = shufflenet_unit(l, channels[2], group, 2 if i == 0 else 1)
            low_res = l

            # reduce high_res channel num by 1x1 conv
            high_res = (LinearWrap(high_res)
                      .Conv2D('conv_low', 32, 1, stride=1)
                      .BatchNorm('bn_low')
                      .LeakyReLU('leaky_low', cfg.leaky_k)())

            high_res = tf.transpose(high_res, [0, 2, 3, 1])
            high_res = tf.space_to_depth(high_res, 2, name="high_res_reshape")
            high_res = tf.transpose(high_res, [0, 3, 1, 2])

            feature = tf.concat([high_res, low_res], axis=1, name="stack_feature")


            # shufflenet-unit for final convs
            with tf.variable_scope('output_group'):
                group = 8
                feature = shufflenet_unit(feature, channels[2] + 32 * 4, group, 1)
                logits = Conv2D('conv_final', feature, cfg.n_boxes * (5 + cfg.n_classes), 1, use_bias=True)

        return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--multi_scale', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        

    model = ShufflenetYolo("NCHW")
    if args.flops:
        cell_h = int(cfg.img_h / cfg.grid_h)
        cell_w = int(cfg.img_w / cfg.grid_w)
        input_desc = [
            InputDesc(tf.uint8, [1, cfg.img_h, cfg.img_w, 3], 'input'),
            InputDesc(tf.float32, [1, cfg.n_boxes, 1, cell_h, cell_w], 'tx'),
            InputDesc(tf.float32, [1, cfg.n_boxes, 1, cell_h, cell_w], 'ty'),
            InputDesc(tf.float32, [1, cfg.n_boxes, 1, cell_h, cell_w], 'tw'),
            InputDesc(tf.float32, [1, cfg.n_boxes, 1, cell_h, cell_w], 'th'),
            InputDesc(tf.float32, [1, cfg.n_boxes, cfg.n_classes, cell_h, cell_w], 'tprob'),
            InputDesc(tf.bool, [1, cfg.n_boxes, cell_h, cell_w], 'spec_mask'),
            InputDesc(tf.float32, [1, cfg.max_box_num, 4], 'truth_box'),
            InputDesc(tf.float32, [1, 3], 'ori_shape'),
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        # assert args.gpu is not None, "Need to specify a list of gpu for training!"
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()
        config = get_config(args, model)
        if args.gpu != None:
            config.nr_tower = len(args.gpu.split(','))

        if args.load:
            config.session_init = get_model_loader(args.load)

        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
