import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]
cfg.learning_rate = [(0, 1e-4),(3, 2e-4),(6, 3e-4),(10, 6e-4),(15, 1e-3), (70, 1e-4), (110, 1e-5)]
cfg.max_epoch = 160

cfg.n_boxes = 5
cfg.n_classes = 20

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 30

cfg.anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

# anchor sizes in [width, height], should be obtained by k-means clustering
# from darknet source code
# anchors = [[0.738768,0.874946],  [2.42204,2.65704],  [4.30971,7.04493],  [10.246,4.59428],  [12.6868,11.8741]]
# for kitti
# anchors = [[1.06593733, 1.03880763], [2.08908397, 4.90636738], [2.35326204, 1.43357071], [4.52926972, 2.49737608], [8.66448722, 4.9158313]]
# for cmdt_109 train
# anchors = [[2.09231069, 3.02696626], [2.89855228, 5.9450238], [4.52319573, 3.7978877], [5.94195853, 6.69752296], [9.05268409, 9.67410767]]

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
	               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
	               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
	               'sofa': 17, 'train': 18, 'tvmonitor': 19}


cfg.train_list = ["voc_2007_train.txt", "voc_2012_train.txt", "voc_2007_val.txt", "voc_2012_val.txt"]
cfg.test_list = "voc_2007_test_without_diff.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = True
cfg.annopath = 'VOCdevkit/VOC2007/Annotations/{:s}.xml'
cfg.imagesetfile = 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'
