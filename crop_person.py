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
    from .reader import Box, box_iou
    from .cfgs.config import cfg
except Exception:
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

def postprocess(predictions, image_path=None, image_shape=None, det_th=None):
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
                if klass_conf[0, n, k, gh, gw] < (det_th or cfg.det_th):
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
    
    return nms_boxes

try:
    from .train import Model as DkModel
    # from .shufflenet_yolo import Model as ShfModel
except Exception:
    from train import Model as DkModel
    # from shufflenet_yolo import Model as ShfModel

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = DkModel()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "spec_mask"],
                                   output_names=["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def generate_pred_images(image_paths, predict_func, output_dir, det_th, enlarge_ratio=1.0):
    for image_idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            continue
        if image_idx % 100 == 0 and image_idx > 0:
            print(str(image_idx))
        print(image_path)
        ori_image = cv2.imread(image_path)

        cvt_color_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(cvt_color_image, (cfg.img_w, cfg.img_h))
        image = np.expand_dims(image, axis=0)
        spec_mask = np.zeros((1, cfg.n_boxes, cfg.img_h // 32, cfg.img_w // 32), dtype=float) == 0
        predictions = predict_func([image, spec_mask])

        boxes = postprocess(predictions, image_path=image_path, det_th=det_th)

        image_name = ntpath.basename(image_path)
        # crop each box and save
        for klass, k_boxes in boxes.items():
            if klass != "person" and klass != "Pedestrian":
                continue
            for box_idx, k_box in enumerate(k_boxes):
                [conf, xmin, ymin, xmax, ymax] = k_box
                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = (xmax - xmin) * enlarge_ratio
                height = (ymax - ymin) * enlarge_ratio
                xmin = np.max([0, int(xcenter - width / 2)])
                ymin = np.max([0, int(ycenter - height / 2)])
                xmax = np.min([ori_image.shape[1] - 1, int(xcenter + width / 2)])
                ymax = np.min([ori_image.shape[0] - 1, int(ycenter + height / 2)])
                if (xmax - xmin) < 32 or (ymax - ymin) < 40 or ((xmax - xmin) >= (ymax - ymin)):
                	continue
                crop_img = ori_image[int(ymin):int(ymax),int(xmin):int(xmax)]

                name_part, img_type = image_name.split('.')
                save_name = name_part + "_" + klass + "_" + str(box_idx) + "." + img_type
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, crop_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--model', help='the model used', default='darknet')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')
    parser.add_argument('--det_th', help='detection threshold', default=0.25)
    parser.add_argument('--input_dir', help='path of the input dir', default=None)
    parser.add_argument('--output_dir', help='directory to save image result', default='output')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    predict_func = get_pred_func(args)

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    image_paths = [os.path.join(args.input_dir, e) for e in os.listdir(args.input_dir)]
            
    print("Number of images to predict: " + str(len(image_paths)))
    generate_pred_images(image_paths, predict_func, args.output_dir, float(args.det_th))
