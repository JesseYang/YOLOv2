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

cfg.anchors = [[1.56661581, 1.27376588], [2.34958704, 2.02082615], [3.46004604, 2.64250903], [4.43872083, 3.78186485], [6.63968814, 6.67205063]]

cfg.max_epoch = 160

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ["draw"]

cfg.classes_num = {'draw': 0}


cfg.train_list = ["draw_train.txt"]
cfg.test_list = "draw_val.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
