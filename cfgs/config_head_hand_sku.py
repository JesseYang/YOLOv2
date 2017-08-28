import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]

cfg.n_boxes = 5
cfg.classes_name =  ["head", "hand", "sku"]
cfg.classes_num = {'head': 0, "hand": 1, "sku": 2}
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


cfg.anchors = [[0.62710315, 0.806326], [0.98381137, 1.67396812], [1.51831471, 1.06341046], [2.01376622, 2.52157951], [2.6802694, 3.84616092]]


# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1


cfg.max_epoch = 160


cfg.train_list = ["head_hand_sku_train.txt"]
cfg.test_list = "head_hand_sku_test.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
