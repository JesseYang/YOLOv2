import os, sys, shutil
import pickle
import numpy as np
import random
from scipy import misc
import six
from six.moves import urllib, range
import copy
import logging
import cv2
import json

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

from tensorpack import *


class Box():
    def __init__(self, p1, p2, p3, p4, mode='XYWH'):
        if mode == 'XYWH':
            # parameters: center_x, center_y, width, height
            self.x = p1
            self.y = p2
            self.w = p3
            self.h = p4
        if mode == "XYXY":
            # parameters: xmin, ymin, xmax, ymax
            self.x = (p1 + p3) / 2
            self.y = (p2 + p4) / 2
            self.w = p3 - p1
            self.h = p4 - p2

def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(box1, box2):
    i = box_intersection(box1, box2)
    u = box1.w * box1.h + box2.w * box2.h - i
    return u

def box_iou(box1, box2):
    return box_intersection(box1, box2) / box_union(box1, box2)

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, affine_trans, use_multi_scale, period):
        self.filename_list = filename_list
        self.use_multi_scale = use_multi_scale
        self.period = period

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        self.affine_trans = affine_trans

    def size(self):
        return len(self.imglist)

    def generate_sample(self, idx, image_height, image_width):
        hflip = False if self.flip == False else (random.random() > 0.5)
        line = self.imglist[idx]

        grid_h = int(image_height / 32)
        grid_w = int(image_width / 32)

        spec_mask = np.zeros((cfg.n_boxes, grid_h, grid_w)).astype(np.float32)

        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])
        s = image.shape
        h, w, c = image.shape

        if self.affine_trans:
            scale = np.random.uniform() / 10. + 1.
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            image = image[offy: (offy + h), offx: (offx + w)]

        if hflip:
            # flip around the vertical axis
            image = cv2.flip(image, flipCode=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        width_rate = image_width * 1.0 / w
        height_rate = image_height * 1.0 / h

        image = cv2.resize(image, (image_width, image_height))

        tx = np.tile(0.5, (cfg.n_boxes, 1, grid_h, grid_w)).astype(np.float32)
        ty = np.tile(0.5, (cfg.n_boxes, 1, grid_h, grid_w)).astype(np.float32)
        tw = np.tile(0, (cfg.n_boxes, 1, grid_h, grid_w)).astype(np.float32)
        th = np.tile(0, (cfg.n_boxes, 1, grid_h, grid_w)).astype(np.float32)
        tprob = np.tile(0, (cfg.n_boxes, cfg.n_classes, grid_h, grid_w)).astype(np.float32)

        truth_box = np.zeros((cfg.max_box_num, 4)).astype(np.float32)
        truth_num = 0

        i = 1
        while i < len(record):
            # reach the max box number
            if truth_num >= cfg.max_box_num:
                break

            # for each ground truth box
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            if self.affine_trans:
                box = np.asarray([xmin, ymin, xmax, ymax])
                box = box * scale
                box[0::2] -= offx
                box[1::2] -= offy
                xmin = np.maximum(0, box[0])
                ymin = np.maximum(1, box[1])
                xmax = np.minimum(w - 1, box[2])
                ymax = np.minimum(h - 1, box[3])
            if hflip:
                xmin = w - 1 - xmin
                xmax = w - 1 - xmax
                tmp = xmin
                xmin = xmax
                xmax = tmp
            class_num = int(record[i + 4])
            i += 5

            # center, width, and height in pixels after resize
            center_w_pixel = (xmin + xmax) * 1.0 / 2 * width_rate
            center_h_pixel = (ymin + ymax) * 1.0 / 2 * height_rate
            box_w_pixel = (xmax - xmin + 1) * width_rate
            box_h_pixel = (ymax - ymin + 1) * height_rate

            # center, width, and height in cells after resize
            eps = 1e-4
            center_w_cell = np.minimum(grid_w - eps, center_w_pixel / 32)
            center_h_cell = np.minimum(grid_h - eps, center_h_pixel / 32)
            box_w_cell = np.minimum(grid_w - eps, box_w_pixel / 32)
            box_h_cell = np.minimum(grid_h - eps, box_h_pixel / 32)
            if box_w_cell < cfg.size_th or box_h_cell < cfg.size_th:
                continue

            # calculate iou between this ground truth box and the anchor boxes
            ious = []
            for anchor_idx, anchor in enumerate(cfg.anchors):
                ious.append(box_iou(Box(0, 0, anchor[0], anchor[1]), Box(0, 0, box_w_cell, box_h_cell)))
            ious = np.asarray(ious)
            truth_idx = np.argmax(ious)

            truth_box[truth_num, :] = np.asarray([center_h_cell, center_w_cell, box_h_cell, box_w_cell])
            truth_num += 1

            if spec_mask[truth_idx, int(center_h_cell), int(center_w_cell)] != 0:
                # already has ground truth box in the same cell with same anchor index
                # has to abandon this ground truth box
                continue

            spec_mask[truth_idx, int(center_h_cell), int(center_w_cell)] = 1.0

            tx[truth_idx, 0, int(center_h_cell), int(center_w_cell)] = center_w_cell - int(center_w_cell)
            ty[truth_idx, 0, int(center_h_cell), int(center_w_cell)] = center_h_cell - int(center_h_cell)
            # b_w = p_w * e^{t_w}
            tw[truth_idx, 0, int(center_h_cell), int(center_w_cell)] = np.log(box_w_cell / cfg.anchors[truth_idx][0])
            # b_h = p_h * e^{t_h}
            th[truth_idx, 0, int(center_h_cell), int(center_w_cell)] = np.log(box_h_cell / cfg.anchors[truth_idx][1])
            tprob[truth_idx, class_num, int(center_h_cell), int(center_w_cell)] = 1

        return [image, tx, ty, tw, th, tprob, spec_mask == 1.0, truth_box, np.asarray(s)]

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        image_num = 0
        if self.use_multi_scale:
            multi_scale_idx = int(random.random() * len(cfg.multi_scale))
            image_height = cfg.multi_scale[multi_scale_idx][0]
            image_width = cfg.multi_scale[multi_scale_idx][1]
        else:
            image_height = cfg.img_h
            image_width = cfg.img_w
        for k in idxs:
            yield self.generate_sample(k, image_height, image_width)
            image_num += 1
            if self.use_multi_scale and image_num % self.period == 0:
                multi_scale_idx = int(random.random() * len(cfg.multi_scale))
                image_height = cfg.multi_scale[multi_scale_idx][0]
                image_width = cfg.multi_scale[multi_scale_idx][1]

    def get_data_idx(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield k

    def reset_state(self):
        super(Data, self).reset_state()

def generate_gt_result(test_path, gt_dir="result_gt", overwrite=True):
    if overwrite == False and os.path.isdir(gt_dir):
        return
    # generate the ground truth files for calculation of average precision
    if overwrite == True and os.path.isdir(gt_dir):
        shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)


    with open(test_path) as f:
        content = f.readlines()

    gt_all = {}

    for line in content:
        record = line.split(' ')
        image_id = os.path.basename(record[0]).split('.')[0] if cfg.gt_format == "voc" else record[0]
        i = 1
        
        gt_cur_img = {}
        while i < len(record):
            class_num = int(record[i + 4])
            class_name = cfg.classes_name[class_num]
            
            if class_name not in gt_cur_img.keys():
                gt_cur_img[class_name] = []
            gt_cur_img[class_name].extend(record[i:i+4])
            
            i += 5
        
        for class_name, boxes in gt_cur_img.items():
            if class_name not in gt_all:
                gt_all[class_name] = []
            d = [image_id]
            d.extend(boxes)
            gt_all[class_name].append(d)
            

    for class_name in cfg.classes_name:
        if class_name in gt_all.keys():
            with open(os.path.join(gt_dir, class_name + ".txt"), 'w') as f:
                for line in gt_all[class_name]:
                    line = [str(ele) for ele in line]
                    f.write(' '.join(line) + '\n')

if __name__ == '__main__':
    pass
