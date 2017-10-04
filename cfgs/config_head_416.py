import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[416, 416]]

cfg.n_boxes = 5
cfg.classes_name =  ["head"]
cfg.classes_num = {}
for idx, class_name in enumerate(cfg.classes_name):
    cfg.classes_num[class_name] = idx
cfg.n_classes = len(cfg.classes_name)

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 20

# cfg.anchors = [[0.90263971, 0.99376649], [1.11438646, 1.28119198], [1.17527658, 1.0079721], [1.43500462, 1.38770567], [2.13707885, 1.9576914]]
cfg.anchors = [[0.97785969, 1.43544048], [1.20725199, 1.85061064], [1.27321629, 1.4559597], [1.55458834, 2.00446375], [2.31516876, 2.82777646]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1


cfg.max_epoch = 160


cfg.train_list = ["head_train.txt"]
cfg.test_list = "head_test.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
