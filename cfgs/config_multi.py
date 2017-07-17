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


cfg.learning_rate = [(0,1e-4),(1,2e-4),(5,3e-4),(10,4e-4),(20,6e-4),(30,1e-3),(40,1e-4),(105,1e-5)]
#cfg.learning_rate = [(0, 1e-4), (20, 2e-4), (50, 3e-4), (100, 4e-4), (150, 5e-4), (200,3e-3), (300,2e-3)]
cfg.max_epoch = 160



cfg.anchors = [[2.28287839, 2.74430641], [2.69750187, 5.08323749], [4.46949321, 3.50367429], [4.78591135, 5.93875096], [7.24740245, 7.34431521]]




##### need param for cmdt data preprocessing
cfg.ROOT_DATA_DIR = "multi_cmdt"
# XML_PATH = 'data_plate/plate.xml'
cfg.test_ratio = 0.1
cfg.train_file_name = "multi_train.txt"
cfg.test_file_name = "multi_test.txt"





# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ["good"]

cfg.classes_num = {'good': 0}


cfg.train_list = ["multi_train.txt"]
cfg.test_list = "multi_test.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
