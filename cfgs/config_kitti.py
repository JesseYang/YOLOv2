import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 640
cfg.img_h = 192
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[128, 416], [160, 544], [192, 640], [224, 736], [256, 832], [288, 960], [320, 1056]]
cfg.learning_rate = [(0,1e-4),(1,2e-4),(5,3e-4),(10,4e-4),(20,6e-4),(30,1e-3),(40,1e-4),(105,1e-5)]
cfg.max_epoch = 160

cfg.n_boxes = 5

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 20

cfg.anchors = [[0.5687777, 0.55382309], [1.11584668, 2.45863625], [1.2741393, 0.82744712], [2.51935263, 1.44592264], [4.60370374, 2.7354634]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
cfg.n_classes = len(cfg.classes_name)

cfg.classes_num = {}
for idx, class_name in enumerate(cfg.classes_name):
    cfg.classes_num[class_name] = idx

cfg.train_list = ["kitti_train.txt"]
cfg.test_list = "kitti_test.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.4

cfg.mAP = True

cfg.gt_from_xml = False
cfg.gt_format = "custom"
