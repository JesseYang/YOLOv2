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
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

try:
    from .cfgs.config import cfg
    from .evaluate import do_python_eval
    from .yolo_utils import YoloModel, get_data, get_config
except Exception:
    from cfgs.config import cfg
    from evaluate import do_python_eval
    from yolo_utils import YoloModel, get_data, get_config

@layer_register(use_scope=None)
def ReLU(x, name=None):
    x = tf.nn.relu(x, name=name)
    return x

class DarknetYoloLite(YoloModel):

    def get_logits(self, image):
        # the network part
        with argscope(Conv2D, nl=tf.identity, use_bias=False), \
             argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            # feature extracotr part
            high_res = (LinearWrap(image)
                      .Conv2D('conv1_1', 32, 3, stride=1)
                      .BatchNorm('bn1_1')
                      .tf.nn.relu('leaky1_1')
                      .MaxPooling('pool1', 2)
                      # 208x208
                      .Conv2D('conv2_1', 64, 3, stride=1)
                      .BatchNorm('bn2_1')
                      .tf.nn.relu('leaky2_1')
                      .MaxPooling('pool2', 2)
                      # 104x104
                      .Conv2D('conv3_1', 128, 3, stride=1)
                      .BatchNorm('bn3_1')
                      .tf.nn.relu('leaky3_1')
                      .Conv2D('conv3_2', 64, 1, stride=1)
                      .BatchNorm('bn3_2')
                      .tf.nn.relu('leaky3_2')
                      .Conv2D('conv3_3', 128, 3, stride=1)
                      .BatchNorm('bn3_3')
                      .tf.nn.relu('leaky3_3')
                      .MaxPooling('pool3', 2)
                      # 52x52
                      .Conv2D('conv4_1', 256, 3, stride=1)
                      .BatchNorm('bn4_1')
                      .tf.nn.relu('leaky4_1')
                      .Conv2D('conv4_2', 128, 1, stride=1)
                      .BatchNorm('bn4_2')
                      .tf.nn.relu('leaky4_2')
                      .Conv2D('conv4_3', 256, 3, stride=1)
                      .BatchNorm('bn4_3')
                      .tf.nn.relu('leaky4_3')
                      .MaxPooling('pool4', 2)
                      # 26x26
                      .Conv2D('conv5_1', 512, 3, stride=1)
                      .BatchNorm('bn5_1')
                      .tf.nn.relu('leaky5_1')
                      .Conv2D('conv5_2', 256, 1, stride=1)
                      .BatchNorm('bn5_2')
                      .tf.nn.relu('leaky5_2')
                      .Conv2D('conv5_3', 512, 3, stride=1)
                      .BatchNorm('bn5_3')
                      .tf.nn.relu('leaky5_3')
                      .Conv2D('conv5_4', 256, 1, stride=1)
                      .BatchNorm('bn5_4')
                      .tf.nn.relu('leaky5_4')
                      .Conv2D('conv5_5', 512, 3, stride=1)
                      .BatchNorm('bn5_5')
                      .tf.nn.relu('leaky5_5')())

            feature = (LinearWrap(high_res)
                      .MaxPooling('pool5', 2)
                      # 13x13
                      .Conv2D('conv6_1', 1024, 3, stride=1)
                      .BatchNorm('bn6_1')
                      .tf.nn.relu('leaky6_1')
                      .Conv2D('conv6_2', 512, 1, stride=1)
                      .BatchNorm('bn6_2')
                      .tf.nn.relu('leaky6_2')
                      .Conv2D('conv6_3', 1024, 3, stride=1)
                      .BatchNorm('bn6_3')
                      .tf.nn.relu('leaky6_3')
                      .Conv2D('conv6_4', 512, 1, stride=1)
                      .BatchNorm('bn6_4')
                      .tf.nn.relu('leaky6_4')
                      .Conv2D('conv6_5', 1024, 3, stride=1)
                      .BatchNorm('bn6_5')
                      .tf.nn.relu('leaky6_5')())

            # new layers part
            low_res = (LinearWrap(feature)
                      .Conv2D('conv7_1', 1024, 3, stride=1)
                      .BatchNorm('bn7_1')
                      .tf.nn.relu('leaky7_1')
                      .Conv2D('conv7_2', 1024, 3, stride=1)
                      .BatchNorm('bn7_2')
                      .tf.nn.relu('leaky7_2')())

            # reduce high_res channel num by 1x1 conv
            high_res = (LinearWrap(high_res)
                      .Conv2D('conv7_3', 64, 1, stride=1)
                      .BatchNorm('bn7_3')
                      .tf.nn.relu('leaky7_3')())

            # concat high_res and low_res
            # tf.space_to_depth requires NHWC format
            if self.data_format == "NCHW":
                high_res = tf.transpose(high_res, [0, 2, 3, 1])
            high_res = tf.space_to_depth(high_res, 2, name="high_res_reshape")
            if self.data_format == "NCHW":
                high_res = tf.transpose(high_res, [0, 3, 1, 2])
            # confirm that the data_format matches with axis
            concat_axis = 1 if self.data_format == "NCHW" else 3
            feature = tf.concat([high_res, low_res], axis=concat_axis, name="stack_feature")

            logits = (LinearWrap(feature)
                   .Conv2D('conv7_4', 1024, 3, stride=1)
                   .BatchNorm('bn7_4')
                   .tf.nn.relu('leaky7_4')
                   .Conv2D('conv7_5', cfg.n_boxes * (5 + cfg.n_classes), 1, stride=1, use_bias=True)())

        logits = tf.identity(logits, 'lite_output')
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
        

    model = DarknetYoloLite()
    if args.flops:
        cell_h = int(cfg.img_h / cfg.grid_h)
        cell_w = int(cfg.img_w / cfg.grid_w)
        input_desc = [
            InputDesc(tf.uint8, [1, 416, 416, 3], 'input'),
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
            config.session_init = SaverRestore(args.load)

        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
