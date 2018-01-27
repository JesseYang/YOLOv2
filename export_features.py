#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

from tensorpack import *

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg


try:
    from .darknet_yolo import DarknetYolo
    from .darknet_yolo_lite import DarknetYoloLite
    from .shufflenet_yolo import ShufflenetYolo
except Exception:
    from darknet_yolo import DarknetYolo
    from darknet_yolo_lite import DarknetYoloLite
    from shufflenet_yolo import ShufflenetYolo


shf_yolo_feat_names = ["network_input",
                       "conv1/output", "pool1/output",
                       "group1/block3/Relu",
                       "group2/block7/Relu",
                       "group3/block3/Relu",
                       "stack_feature",
                       "output_group/conv_final/output",
                       "pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"]

darknet_yolo_feat_names = ["network_input",
                           "conv1_1/output", "bn1_1/output", "leaky1_1/output", "pool1/output",
                           "conv2_1/output", "bn2_1/output", "leaky2_1/output", "pool2/output",
                           "conv3_1/output", "bn3_1/output", "leaky3_1/output",
                           "conv3_2/output", "bn3_2/output", "leaky3_2/output",
                           "conv3_3/output", "bn3_3/output", "leaky3_3/output", "pool3/output",
                           "conv4_1/output", "bn4_1/output", "leaky4_1/output",
                           "conv4_2/output", "bn4_2/output", "leaky4_2/output",
                           "conv4_3/output", "bn4_3/output", "leaky4_3/output", "pool4/output",
                           "conv5_1/output", "bn5_1/output", "leaky5_1/output",
                           "conv5_2/output", "bn5_2/output", "leaky5_2/output",
                           "conv5_3/output", "bn5_3/output", "leaky5_3/output",
                           "conv5_4/output", "bn5_4/output", "leaky5_4/output",
                           "conv5_5/output", "bn5_5/output", "leaky5_5/output", "pool5/output",
                           "conv6_1/output", "bn6_1/output", "leaky6_1/output",
                           "conv6_2/output", "bn6_2/output", "leaky6_2/output",
                           "conv6_3/output", "bn6_3/output", "leaky6_3/output",
                           "conv6_4/output", "bn6_4/output", "leaky6_4/output",
                           "conv6_5/output", "bn6_5/output", "leaky6_5/output",
                           "conv7_1/output", "bn7_1/output", "leaky7_1/output",
                           "conv7_2/output", "bn7_2/output", "leaky7_2/output",
                           "conv7_3/output", "bn7_3/output", "leaky7_3/output",
                           "stack_feature",
                           "conv7_4/output", "bn7_4/output", "leaky7_4/output",
                           "conv7_5/output",
                           "pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"]

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    if args.model == "darknet":
        model = DarknetYolo()
        feat_names = darknet_yolo_feat_names
    else:
        model = ShufflenetYolo("NCHW")
        feat_names = shf_yolo_feat_names
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=feat_names)

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def do_export(input_path, output_path, predict_func):
    ori_image = cv2.imread(input_path)
    cvt_clr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(cvt_clr_image, (cfg.img_w, cfg.img_h))
    image = np.expand_dims(image, axis=0)

    predictions = predict_func(image)

    if args.model == "darknet":
        feat_names = darknet_yolo_feat_names
    else:
        feat_names = shf_yolo_feat_names

    feat_dict = { }
    for feat_idx, feat_name in enumerate(feat_names):
        key_name = feat_name.split('/')[0]
        feat_dict[key_name] = predictions[feat_idx][0]

    np.save(output_path, feat_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.', required=True)
    parser.add_argument('--model', choices=['darknet', 'shufflenet'], default='shufflenet')
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='features.npy')
    args = parser.parse_args()


    predict_func = get_pred_func(args)

    do_export(args.input_path, args.output_path, predict_func)
