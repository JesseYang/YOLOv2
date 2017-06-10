#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pdb
import shutil
import numpy as np
from scipy import misc
import argparse
import json
import cv2

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

from tensorpack import *

from reader import Box, box_iou
from cfgs.config import cfg

def non_maximum_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    conf = boxes[:,0]
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        union = area[idxs[:last]] + area[idxs[last]] - intersection
 
        # compute the ratio of overlap
        # overlap = (w * h) / area[idxs[:last]]
        overlap = intersection / union
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("float")

def postprocess(predictions, image_path=None, image_shape=None):
    if image_path != None:
        ori_image = cv2.imread(image_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image_shape = ori_image.shape
    ori_height = image_shape[0]
    ori_width = image_shape[1]

    [pred_x, pred_y, pred_w, pred_h, pred_conf, pred_prob] = predictions

    _, box_n, klass_num, grid_h, grid_w = pred_prob.shape

    pred_conf_tile = np.tile(pred_conf, (1, 1, klass_num, 1, 1))
    klass_conf = pred_prob * pred_conf_tile

    width_rate = ori_width / float(cfg.img_w)
    height_rate = ori_height / float(cfg.img_h)

    boxes = {}
    for n in range(box_n):
        for gh in range(grid_h):
            for gw in range(grid_w):

                k = np.argmax(klass_conf[0, n, :, gh, gw])
                if klass_conf[0, n, k, gh, gw] < cfg.det_th:
                    continue

                anchor = cfg.anchors[n]
                w = pred_w[0, n, 0, gh, gw]
                h = pred_h[0, n, 0, gh, gw]
                x = pred_x[0, n, 0, gh, gw]
                y = pred_y[0, n, 0, gh, gw]

                center_w_cell = gw + x
                center_h_cell = gh + y
                box_w_cell = np.exp(w) * anchor[0]
                box_h_cell = np.exp(h) * anchor[1]

                center_w_pixel = center_w_cell * 32
                center_h_pixel = center_h_cell * 32
                box_w_pixel = box_w_cell * 32
                box_h_pixel = box_h_cell * 32

                xmin = float(center_w_pixel - box_w_pixel // 2)
                ymin = float(center_h_pixel - box_h_pixel // 2)
                xmax = float(center_w_pixel + box_w_pixel // 2)
                ymax = float(center_h_pixel + box_h_pixel // 2)
                xmin = np.max([xmin, 0]) * width_rate
                ymin = np.max([ymin, 0]) * height_rate
                xmax = np.min([xmax, float(cfg.img_w)]) * width_rate
                ymax = np.min([ymax, float(cfg.img_h)]) * height_rate

                klass = cfg.classes_name[k]
                if klass not in boxes.keys():
                    boxes[klass] = []

                box = [klass_conf[0, n, k, gh, gw], xmin, ymin, xmax, ymax]

                boxes[klass].append(box)

    # do non-maximum-suppresion
    nms_boxes = {}
    if cfg.nms == True:
        for klass, k_boxes in boxes.items():
            k_boxes = non_maximum_suppression(k_boxes, cfg.nms_th)
            nms_boxes[klass] = k_boxes
    else:
        nms_boxes = boxes

    
    # draw result
    if image_path != None:
        colors = [(255,0,0), (0,255,0), (0,0,255),
                  (255,255,0), (255,0,255), (0,255,255),
                  (122,0,0), (0,122,0), (0,0,122),
                  (122,122,0), (122,0,122), (0,122,122)]

        text_colors = [(0,255,255), (255,0,255), (255,255,0),
                      (0,0,255), (0,255,0), (255,0,0),
                      (0,122,122), (122,0,122), (122,122,0),
                      (0,0,122), (0,122,0), (122,0,0)]

        image_result = np.copy(ori_image)
        k_idx = 0
        for klass, k_boxes in nms_boxes.items():
            for k_box in k_boxes:

                [conf, xmin, ymin, xmax, ymax] = k_box

                label_height = 14
                label_width = len(klass) * 10
     
                cv2.rectangle(image_result,
                              (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)),
                              colors[k_idx % len(colors)],
                              3)
                cv2.rectangle(image_result,
                              (int(xmin) - 2, int(ymin) - label_height),
                              (int(xmin) + label_width, int(ymin)),
                              colors[k_idx % len(colors)],
                              -1)
                cv2.putText(image_result,
                            klass,
                            (int(xmin), int(ymin) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            text_colors[k_idx % len(text_colors)])
            k_idx += 1
    else:
        image_result = None

    return nms_boxes, image_result

from train import Model

def get_pred_func(model_path):
    sess_init = SaverRestore(model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "spec_mask"],
                                   output_names=["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def predict_image(image_path, predict_func):
    ori_image = cv2.imread(image_path)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(ori_image, (cfg.img_h, cfg.img_w))
    image = np.expand_dims(image, axis=0)
    spec_mask = np.zeros((1, cfg.n_boxes, cfg.img_w // 32, cfg.img_h // 32), dtype=float) == 0
    predictions = predict_func([image, spec_mask])

    pred_results, img_result = postprocess(predictions, image_path=image_path)

    return pred_results, img_result

def generate_pred_result(test_path, pred_dir="result_pred"):
    if os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)

    with open(test_path) as f:
        content = f.readlines()

    predict_func = get_pred_func(args.model_path)

    for class_name in cfg.classes_name:
        with open(os.path.join(pred_dir, class_name + ".txt"), 'w') as f:
            continue

    print("Number of images to predict: " + str(len(content)))
            
    for image_idx, line in enumerate(content):
        if image_idx % 100 == 0 and image_idx > 0:
            print(str(image_idx))
        
        record = line.split(' ')
        image_path = record[0]
        image_id = os.path.basename(image_path).split('.')[0]

        ori_image = cv2.imread(image_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(ori_image, (cfg.img_h, cfg.img_w))
        image = np.expand_dims(image, axis=0)
        spec_mask = np.zeros((1, cfg.n_boxes, cfg.img_w // 32, cfg.img_h // 32), dtype=float) == 0
        predictions = predict_func([image, spec_mask])

        pred_results, img_result = postprocess(predictions, image_path=image_path)

        for class_name in pred_results.keys():
            with open(os.path.join(pred_dir, class_name + ".txt"), 'a') as f:
                for box in pred_results[class_name]:
                    record = [image_id]
                    record.extend(box)
                    record = [str(ele) for ele in record]
                    f.write(' '.join(record) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--test_path', help='path of the test file', default='voc_2007_test.txt')
    args = parser.parse_args()

    generate_pred_result(args.test_path)
