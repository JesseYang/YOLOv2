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
import random



def copy_file():
    root = '/home/user/VideoText/DEMO'
    files = ['output20171204-112257', 'output20171204-112551', 'output20171204-112955', 'output20171204-113325', 'output20171204-113636',\
    'output20171204-113937', 'output20171204-114137', 'output20171204-114807', 'output20171204-121711', 'output20171204-115235']
    
    dir_ = os.path.join(root, 'detect', 'act_frames')
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

    for file in files:
        frames = os.path.join(root, file, 'extract_frames')
        imgs = os.listdir(frames)
        for img in imgs:
            name = file + img

            shutil.copy(os.path.join(frames, img), os.path.join(dir_, name))
        
def generate_extract_txt():
    root_data = '/home/user/VideoText/DEMO/detect/act_frames'
    imgs = os.listdir(root_data)
    print(len(imgs))
    result = open("extracted_frame.txt", 'w')
    for img in imgs:
        result.write(os.path.join(root_data, img) + "\n")


def val_kitti():
    file = 'test_kitti_train.txt'
    with open(file) as f:
        labels = f.readlines()
    print("num ", len(labels))

    for label in labels:
        img_path = label.split(' ')[0]
        print(img_path)
        # bbox = [int(float(ele)) for ele in label.strip().split(' ')[1:]]
        bbox = [ele for ele in label.strip().split(' ')[1:]]
        print(bbox)
        bbox = np.array(bbox, dtype = np.float32).reshape(-1,4)
        img = cv2.imread(img_path)
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in bbox:
            cv2.rectangle(img_color, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        # h, w, c = img.shape
        # 
        # neg_num = 0 

        cv2.imwrite(os.path.join('kitti_dir', str(uuid.uuid4()) + ".jpg"), img_color)
if __name__ == '__main__':
    # copy_file()
    # generate_extract_txt()
    val_kitti()