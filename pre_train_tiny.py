#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: inceptionv3.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug


import multiprocessing

TOTAL_BATCH_SIZE = 128
INPUT_SHAPE = 224


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        image = tf.transpose(image, [0, 3, 1, 2])

        with argscope(Conv2D, nl=tf.identity, use_bias=False), \
             argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format="NCHW"):
            logits = (LinearWrap(image)
                      .Conv2D('conv1', 16, 3, stride=1)
                      .BatchNorm('bn1')
                      .LeakyReLU('leaky1', 0.1)
                      .MaxPooling('pool1', 2)
                      # 112
                      .Conv2D('conv2', 32, 3, stride=1)
                      .BatchNorm('bn2')
                      .LeakyReLU('leaky2', 0.1)
                      .MaxPooling('pool2', 2)
                      # 56
                      .Conv2D('conv3', 64, 3, stride=1)
                      .BatchNorm('bn3')
                      .LeakyReLU('leaky3', 0.1)
                      .MaxPooling('pool3', 2)
                      # 28
                      .Conv2D('conv4', 128, 1, stride=1)
                      .BatchNorm('bn4')
                      .LeakyReLU('leaky4', 0.1)
                      .MaxPooling('pool4', 2)
                      # 14
                      .Conv2D('conv5', 256, 3, stride=1)
                      .BatchNorm('bn5')
                      .LeakyReLU('leaky5', 0.1)
                      .MaxPooling('pool5', 2)
                      # 7
                      .Conv2D('conv6', 512, 3, stride=1)
                      .BatchNorm('bn6')
                      .LeakyReLU('leaky6', 0.1)
                      .MaxPooling('pool6', 2, padding="SAME")
                      # 4
                      .Conv2D('conv7', 1024, 3, stride=1)
                      .BatchNorm('bn7')
                      .LeakyReLU('leaky7', 0.1)
                      # output
                      .Conv2D('conv8', 1000, 1, stride=1)
                      .LeakyReLU('leaky7', 0.1)
                      .GlobalAvgPooling('gap')())

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # loss = tf.reduce_mean(loss, name='xentropy-loss')
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
        loss = tf.identity(loss, name='xentropy-loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        wd_cost = regularize_cost('.*/W', l2_regularizer(5e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test):
    # return FakeData([[64, 224,224,3],[64]], 1000, random=False, dtype='uint8')
    isTrain = train_or_test == 'train'

    datadir = args.data
    ds = dataset.ILSVRC12(datadir, train_or_test,
                          shuffle=True if isTrain else False, dir_structure='original')
    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            """
            crop 8%~100% of the original image
            See `Going Deeper with Convolutions` by Google.
            """
            def _augment(self, img, _):
                h, w = img.shape[:2]
                area = h * w
                for _ in range(10):
                    targetArea = self.rng.uniform(0.08, 1.0) * area
                    aspectR = self.rng.uniform(0.75, 1.333)
                    ww = int(np.sqrt(targetArea * aspectR))
                    hh = int(np.sqrt(targetArea / aspectR))
                    if self.rng.uniform() < 0.5:
                        ww, hh = hh, ww
                    if hh <= h and ww <= w:
                        x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                        y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                        out = img[y1:y1 + hh, x1:x1 + ww]
                        out = cv2.resize(out, (INPUT_SHAPE, INPUT_SHAPE), interpolation=cv2.INTER_CUBIC)
                        return out
                out = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE), interpolation=cv2.INTER_CUBIC)
                return out

        augmentors = [
            Resize(),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045],
                                 eigvec=[[-0.5675, 0.7192, 0.4009],
                                         [-0.5808, -0.0045, -0.8140],
                                         [-0.5836, -0.6948, 0.4203]])
                                 ]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(512),
            imgaug.CenterCrop((INPUT_SHAPE, INPUT_SHAPE)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(20, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds

def get_config(debug):
    dataset_train = get_data('train')
    dataset_val = get_data('val')

    sess = SessionCreatorAdapter(NewSessionCreator(), lambda sess: tf_debug.LocalCLIDebugWrapperSession(sess)) if debug else None
    return TrainConfig(
        session_creator=sess,
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
            # HyperParamSetterWithFunc('learning_rate',
            #                          lambda e, x: 1e-3 * (1 - e * 1.0 / 12) ** 4 ),
            HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: 1e-1 * (1 - e * 1.0 / 160) ** 4 ),
        ],
        model=Model(),
        steps_per_epoch=10000,
        max_epoch=160,
        # max_epoch=10,
    )


def eval_on_ILSVRC12(model_file, data_dir):
    ds = get_data('val')
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_file),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='ILSVRC')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.eval:
        BATCH_SIZE = 128    # something that can run on one gpu
        eval_on_ILSVRC12(args.load, args.data)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    logger.auto_set_dir()
    config = get_config(args.debug)
    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()

