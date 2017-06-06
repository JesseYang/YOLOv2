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

cfg.anchors = [[1.33119195, 1.15934275], [2.06045752, 1.6672473], [2.8275166, 2.38338939], [4.10338642, 3.1694666], [6.50620722, 6.22560465]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ["draw"]

cfg.classes_num = {'draw': 0}


cfg.train_list = ["draw_train.txt"]
cfg.test_list = "draw_val.txt"

cfg.det_th = 0.3
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.4
