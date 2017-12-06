import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 384
cfg.img_h = 512
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]

cfg.n_boxes = 5
cfg.n_classes = 3

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 10

#cfg.anchors = [[4.00939542, 4.76900498], [4.73943582, 5.91242], [5.45915033, 6.60725228], [6.3061683, 7.99082797], [7.84313725, 10.28370098]]
cfg.anchors = [[2.35413089, 1.80181272], [5.66890445, 9.15139471], [5.85887638, 3.98212013],  [6.80131994, 11.29153614], [8.52137702, 13.33949497]]

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ['text_area', 'figure', 'table']
cfg.batch_size = 8
cfg.classes_num = {'text_area': 0,
                   'figure': 1,
                   'table': 2
                  }


cfg.train_list = ["doc_train.txt"]
cfg.test_list = "doc_test.txt"

cfg.det_th = 0.01
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
cfg.gt_format = "custom"

cfg.max_epoch = 200
