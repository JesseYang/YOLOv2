import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]

cfg.n_boxes = 5
cfg.n_classes = 1

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 10

cfg.anchors = [[1.38278449, 1.47548495], [2.17105324, 1.74288296], [2.46725112, 1.29884392], [3.23908308, 2.89054875], [3.63421207, 1.58081144]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ["plate"]

cfg.classes_num = {'plate': 0}


cfg.train_list = ["plate_temp.txt"]
cfg.test_list = "plate_temp.txt"

cfg.det_th = 0.3
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.4

cfg.mAP = False
