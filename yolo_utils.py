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
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

try:
    from .cfgs.config import cfg
    from .reader import Data, generate_gt_result
    from .evaluate import do_python_eval
    from .utils import postprocess
except Exception:
    from cfgs.config import cfg
    from reader import Data, generate_gt_result
    from evaluate import do_python_eval
    from utils import postprocess

class YoloModel(ModelDesc):

    def __init__(self, data_format="NHWC", multi_scale=False):
        super(YoloModel, self).__init__()
        self.data_format = data_format
        self.multi_scale = multi_scale

    def _get_inputs(self):
        if self.multi_scale == False:
            cell_h = int(cfg.img_h / cfg.grid_h)
            cell_w = int(cfg.img_w / cfg.grid_w)
            return [InputDesc(tf.uint8, [None, cfg.img_h, cfg.img_w, 3], 'input'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, cell_h, cell_w], 'tx'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, cell_h, cell_w], 'ty'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, cell_h, cell_w], 'tw'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, cell_h, cell_w], 'th'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, cfg.n_classes, cell_h, cell_w], 'tprob'),
                    InputDesc(tf.bool, [None, cfg.n_boxes, cell_h, cell_w], 'spec_mask'),
                    InputDesc(tf.float32, [None, cfg.max_box_num, 4], 'truth_box'),
                    InputDesc(tf.float32, [None, 3], 'ori_shape'),
                    ]
        else:
            return [InputDesc(tf.uint8, [None, None, None, 3], 'input'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, None, None], 'tx'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, None, None], 'ty'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, None, None], 'tw'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, 1, None, None], 'th'),
                    InputDesc(tf.float32, [None, cfg.n_boxes, cfg.n_classes, None, None], 'tprob'),
                    InputDesc(tf.bool, [None, cfg.n_boxes, None, None], 'spec_mask'),
                    InputDesc(tf.float32, [None, cfg.max_box_num, 4], 'truth_box'),
                    InputDesc(tf.float32, [None, 3], 'ori_shape'),
                    ]

    def cal_multi_multi_iou(self, boxes1, boxes2):
        """
        Calculate ious between boxes1, and boxes2

        Args:
            boxes1 (tf.Tensor): a 5D (batch x n_boxes x grid_w x grid_h x 4) tensor. Length of the last dimension is 4 (x, y, w, h)
            boxes2 (tf.Tensor): a 5D (batch x n_boxes x grid_w x grid_h x 4) tensor. Length of the last dimension is 4 (x, y, w, h)

        Returns:
            4D tf.Tensor (batch x n_boxes x grid_w x grid_h).

        """
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2],
                          axis=4)

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2],
                          axis=4)

        #calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, :, 0:2], boxes2[:, :, :, :, 0:2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:],  boxes2[:, :, :, :, 2:])

        #intersection
        intersection = rd - lu 

        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        mask = tf.cast(intersection[:, :, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, :, 1] > 0, tf.float32)
        
        inter_square = mask * inter_square
        
        #calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
        
        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def cal_multi_one_iou(self, b_pred, b_one_truth):
        b_one_truth = tf.tile(b_one_truth, [1, cfg.n_boxes * self.grid_h * self.grid_w])
        b_one_truth = tf.reshape(b_one_truth, (-1, cfg.n_boxes, self.grid_h, self.grid_w, 4))
        iou = self.cal_multi_multi_iou(b_pred, b_one_truth)
        return iou

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of cfg.img_hxcfg.img_w in ``self.data_format``
        Returns:
            cfg.img_h/32 x cfg.img_w/32 logits in ``self.data_format``
        """

    def _build_graph(self, inputs):
        image, tx, ty, tw, th, tprob, spec_mask, truth_box, ori_shape = inputs
        self.batch_size = tf.shape(image)[0]
        self.grid_h = tf.cast(tf.shape(image)[1] / 32, dtype=tf.int32)
        self.grid_w = tf.cast(tf.shape(image)[2] / 32, dtype=tf.int32)
        self.unseen_scale = get_scalar_var('unseen_scale', 0, summary=True)

        spec_indicator = tf.reshape(tf.cast(spec_mask, tf.float32), (-1, cfg.n_boxes, 1, self.grid_h, self.grid_w))

        coord_scale = spec_indicator * cfg.coord_scale + (1 - spec_indicator) * self.unseen_scale
        conf_scale = tf.ones((self.batch_size, cfg.n_boxes, 1, self.grid_h, self.grid_w)) * cfg.noobject_scale
        class_scale = spec_indicator * cfg.class_scale
        class_scale = tf.tile(class_scale, [1, 1, cfg.n_classes, 1, 1], name="class_scale")

        tf.summary.image('input-image', image, max_outputs=3)

        image = tf.cast(image, tf.float32) * (1.0 / 255)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        image = tf.identity(image, name='network_input')

        logits = self.get_logits(image)

        # the loss part, confirm that logits is NCHW format
        if self.data_format == "NHWC":
            logits = tf.transpose(logits, [0, 3, 1, 2])
        pred = tf.reshape(logits, (-1, cfg.n_boxes, cfg.n_classes + 5, self.grid_h, self.grid_w))
        # each predictor has dimension: batch x n_boxes x value x grid_w x grid_h
        # for x, y, w, h, and conf, value is 1; for prob, value is n_classes
        x, y, w, h, conf, prob = tf.split(pred, num_or_size_splits=[1, 1, 1, 1, 1, cfg.n_classes], axis=2)

        x = tf.sigmoid(x, name="pred_x")
        y = tf.sigmoid(y, name="pred_y")
        w = tf.identity(w, name="pred_w")
        h = tf.identity(h, name="pred_h")
        conf = tf.sigmoid(conf, name="pred_conf")

        x_loss = tf.multiply(tf.square(tf.subtract(x, tx)), coord_scale)
        y_loss = tf.multiply(tf.square(tf.subtract(y, ty)), coord_scale)
        w_loss = tf.multiply(tf.square(tf.subtract(w, tw)), coord_scale)
        h_loss = tf.multiply(tf.square(tf.subtract(h, th)), coord_scale)

        x_loss = tf.div(tf.reduce_mean(tf.reduce_sum(x_loss, [1, 2, 3, 4])), 2, name="x_loss")
        y_loss = tf.div(tf.reduce_mean(tf.reduce_sum(y_loss, [1, 2, 3, 4])), 2, name="y_loss")
        w_loss = tf.div(tf.reduce_mean(tf.reduce_sum(w_loss, [1, 2, 3, 4])), 2, name="w_loss")
        h_loss = tf.div(tf.reduce_mean(tf.reduce_sum(h_loss, [1, 2, 3, 4])), 2, name="h_loss")

        if cfg.n_classes > 1:
            prob = tf.nn.softmax(prob, 2)
            prob = tf.reshape(prob, (-1, cfg.n_boxes, cfg.n_classes, self.grid_h, self.grid_w), name="pred_prob")
            p_loss = tf.multiply(tf.square(tf.subtract(prob, tprob)), class_scale)
            p_loss = tf.div(tf.reduce_mean(tf.reduce_sum(p_loss, [1, 2, 3, 4])), 2, name="p_loss")
        else:
            prob = tf.ones((self.batch_size, cfg.n_boxes, cfg.n_classes, self.grid_h, self.grid_w), name="pred_prob")


        # for c_loss, the truth value tconf is the iou between the predictor box and ground truth box
        # calculate tconf
        grid_ary_x = tf.cast(tf.range(self.grid_w), tf.float32)
        grid_tensor_x = tf.tile(grid_ary_x, [self.batch_size * cfg.n_boxes * self.grid_h])
        grid_tensor_x = tf.reshape(grid_tensor_x, (-1, cfg.n_boxes, self.grid_h, self.grid_w))

        grid_ary_y = tf.cast(tf.range(self.grid_h), tf.float32)
        grid_tensor_y = tf.tile(grid_ary_y, [self.batch_size * cfg.n_boxes * self.grid_w])
        grid_tensor_y = tf.reshape(grid_tensor_y, (-1, cfg.n_boxes, self.grid_w, self.grid_h))
        grid_tensor_y = tf.transpose(grid_tensor_y, (0, 1, 3, 2))

        anchor_ary = tf.cast(tf.constant(cfg.anchors), tf.float32)

        anchor_ary_width = anchor_ary[:, 0]
        anchor_tensor_width = tf.tile(anchor_ary_width, [self.batch_size * self.grid_h * self.grid_w])
        anchor_tensor_width = tf.reshape(anchor_tensor_width, (self.batch_size, self.grid_h, self.grid_w, cfg.n_boxes))
        anchor_tensor_width = tf.transpose(anchor_tensor_width, (0, 3, 1, 2))

        anchor_ary_height = anchor_ary[:, 1]
        anchor_tensor_height = tf.tile(anchor_ary_height, [self.batch_size * self.grid_h * self.grid_w])
        anchor_tensor_height = tf.reshape(anchor_tensor_height, (self.batch_size, self.grid_h, self.grid_w, cfg.n_boxes))
        anchor_tensor_height = tf.transpose(anchor_tensor_height, (0, 3, 1, 2))

        # b_pred is the predictor box, the unit is "cell"
        b_x = tf.reshape(x, (-1, cfg.n_boxes, self.grid_h, self.grid_w)) + grid_tensor_x
        b_y = tf.reshape(y, (-1, cfg.n_boxes, self.grid_h, self.grid_w)) + grid_tensor_y
        b_w = tf.reshape(tf.exp(w), (-1, cfg.n_boxes, self.grid_h, self.grid_w)) * anchor_tensor_width
        b_h = tf.reshape(tf.exp(h), (-1, cfg.n_boxes, self.grid_h, self.grid_w)) * anchor_tensor_height
        b_pred = tf.stack([b_x, b_y, b_w, b_h], axis=4, name="pred_boxes")

        # b_truth is the grouth box, the unit is "cell". for those locations without truth boxes, b_truth has the standard anchor
        b_tx = tf.reshape(tx, (-1, cfg.n_boxes, self.grid_h, self.grid_w)) + grid_tensor_x
        b_ty = tf.reshape(ty, (-1, cfg.n_boxes, self.grid_h, self.grid_w)) + grid_tensor_y
        b_tw = tf.reshape(tf.exp(tw), (-1, cfg.n_boxes, self.grid_h, self.grid_w)) * anchor_tensor_width
        b_th = tf.reshape(tf.exp(th), (-1, cfg.n_boxes, self.grid_h, self.grid_w)) * anchor_tensor_height
        b_truth = tf.stack([b_tx, b_ty, b_tw, b_th], axis=4, name="truth_boxes")

        # effective elements in tconf: where there is a truth box
        tconf = self.cal_multi_multi_iou(b_pred, b_truth)
        tconf = tf.reshape(tconf, (-1, cfg.n_boxes, 1, self.grid_h, self.grid_w))
        # for thoes without a truth box, set tconf as 0
        tconf = spec_indicator * tconf 

        iou_list = []
        for i in range(cfg.max_box_num):
            iou_list.append(self.cal_multi_one_iou(b_pred, truth_box[:, i, :]))

        best_iou = tf.reduce_max(tf.stack(iou_list, axis=0), axis=0)

        high_iou_mask = best_iou > cfg.threshold
        high_iou_mask = tf.cast(high_iou_mask, tf.float32)
        high_iou_mask = tf.reshape(high_iou_mask, (-1, cfg.n_boxes, 1, self.grid_h, self.grid_w))
        # for those locations where predicted box has an iou greater then threshold with a truth box, set conf_scale to 0
        conf_scale = (1 - high_iou_mask) * conf_scale 
        # for those locations having truth boxes, the conf_scale is set as object_scale
        conf_scale = (1 - spec_indicator) * conf_scale  + spec_indicator * cfg.object_scale 

        conf_scale = tf.stop_gradient(conf_scale)
        tconf = tf.stop_gradient(tconf)

        c_loss = tf.multiply(tf.square(tf.subtract(conf, tconf)), conf_scale)
        c_loss = tf.div(tf.reduce_mean(tf.reduce_sum(c_loss, [1, 2, 3, 4])), 2, name="c_loss")

        coord_loss = tf.add_n([x_loss, y_loss, w_loss, h_loss], name="coord_loss")

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)
        if cfg.n_classes > 1:
            loss = tf.add_n([coord_loss, c_loss, p_loss], name='loss')
            add_moving_summary(x_loss, y_loss, w_loss, h_loss, c_loss, p_loss, loss, wd_cost)
        else:
            loss = tf.add_n([coord_loss, c_loss], name='loss')
            add_moving_summary(x_loss, y_loss, w_loss, h_loss, c_loss, loss, wd_cost)

        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

class CalMAP(Inferencer):
    def __init__(self, test_path):
        self.names = ["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob", "ori_shape", "loss"]
        self.test_path = test_path
        self.gt_dir = "result_gt"
        if os.path.isdir(self.gt_dir):
            shutil.rmtree(self.gt_dir)

        self.pred_dir = "result_pred/"
        if os.path.isdir(self.pred_dir):
            shutil.rmtree(self.pred_dir)
        os.mkdir(self.pred_dir)

        with open(self.test_path) as f:
            content = f.readlines()

        self.image_path_list = []
        for line in content:
            self.image_path_list.append(line.split(' ')[0])

        self.cur_image_idx = 0

    def _get_fetches(self):
        return self.names

    def _before_inference(self):
        # if the "result_gt" dir does not exist, generate it from the data_set
        generate_gt_result(self.test_path, self.gt_dir, overwrite=False)
        self.results = { }
        self.loss = []
        self.cur_image_idx = 0

    def _on_fetches(self, output):
        self.loss.append(output[7])
        output = output[0:7]
        for i in range(output[0].shape[0]):
            # for each ele in the batch
            image_path = self.image_path_list[self.cur_image_idx]
            self.cur_image_idx += 1
            image_id = os.path.basename(image_path).split('.')[0] if cfg.gt_format == "voc" else image_path

            cur_output = [ele[i] for ele in output]
            predictions = [np.expand_dims(ele, axis=0) for ele in cur_output[0:6]]
            image_shape = cur_output[6]

            pred_results = postprocess(predictions, image_shape=image_shape)
            for class_name in pred_results.keys():
                if class_name not in self.results.keys():
                    self.results[class_name] = []
                for box in pred_results[class_name]:
                    record = [image_id]
                    record.extend(box)
                    record = [str(ele) for ele in record]
                    self.results[class_name].append(' '.join(record))

    def _after_inference(self):
        # write the result to file
        for class_name in self.results.keys():
            with open(os.path.join(self.pred_dir, class_name + ".txt"), 'wt') as f:
                for record in self.results[class_name]:
                    f.write(record + '\n')
        # calculate the mAP based on the predicted result and the ground truth
        mAP = do_python_eval(self.pred_dir)
        return { "mAP": mAP, "test_loss": np.mean(self.loss) }


def get_data(train_or_test, multi_scale, batch_size):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, flip=isTrain, affine_trans=isTrain, use_multi_scale=isTrain and multi_scale, period=batch_size*10)

    if isTrain:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain and multi_scale == False:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds


def get_config(args, model):
    if args.gpu != None:
        NR_GPU = len(args.gpu.split(','))
        batch_size = int(args.batch_size) // NR_GPU
    else:
        batch_size = int(args.batch_size)

    ds_train = get_data('train', args.multi_scale, batch_size)
    ds_test = get_data('test', False, batch_size)

    callbacks = [
      ModelSaver(),


      ScheduledHyperParamSetter('learning_rate',
                                cfg.lr_schedule),
                                # [(0, 1e-4), (3, 2e-4), (6, 3e-4), (10, 6e-4), (15, 1e-3), (60, 1e-4), (90, 1e-5)]),
      ScheduledHyperParamSetter('unseen_scale',
                                [(0, cfg.unseen_scale), (cfg.unseen_epochs, 0)]),
      HumanHyperParamSetter('learning_rate'),
    ]
    if cfg.mAP == True:
        callbacks.append(PeriodicTrigger(InferenceRunner(ds_test, [CalMAP(cfg.test_list)]),
                                         every_k_epochs=3))
    if args.debug:
      callbacks.append(HookToCallback(tf_debug.LocalCLIDebugHook()))
    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=1810,
        max_epoch=cfg.max_epoch,
    )

