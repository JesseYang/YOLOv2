from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32

cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]

cfg.n_boxes = 5
cfg.n_classes = 80

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 30

'''
k-means clustering pascal anchor points (original coordinates)
Found at iteration 15 with best average IoU: 0.5075781190660665 
[[  0.51797281   0.81316624]
 [  1.79211253   1.974359  ]
 [  2.32512963   4.99556911]
 [  5.40091498   6.45247529]
 [ 10.33327151   9.49567865]]
'''

cfg.anchors = [[0.51797281, 0.81316624], [1.79211253, 1.974359], [2.32512963, 4.99556911], [5.40091498, 6.45247529], [10.33327151, 9.49567865]]

cfg.classes_name =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cfg.classes_num = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 26, 'umbrella': 27, 'handbag': 30, 'tie': 31, 'suitcase': 32, 'frisbee': 33, 'skis': 34, 'snowboard': 35, 'sports ball': 36, 'kite': 37, 'baseball bat': 38, 'baseball glove': 39, 'skateboard': 40, 'surfboard': 41, 'tennis racket': 42, 'bottle': 43, 'wine glass': 45, 'cup': 46, 'fork': 47, 'knife': 48, 'spoon': 49, 'bowl': 50, 'banana': 51, 'apple': 52, 'sandwich': 53, 'orange': 54, 'broccoli': 55, 'carrot': 56, 'hot dog': 57, 'pizza': 58, 'donut': 59, 'cake': 60, 'chair': 61, 'couch': 62, 'potted plant': 63, 'bed': 64, 'dining table': 66, 'toilet': 69, 'tv': 71, 'laptop': 72, 'mouse': 73, 'remote': 74, 'keyboard': 75, 'cell phone': 76, 'microwave': 77, 'oven': 78, 'toaster': 79, 'sink': 80, 'refrigerator': 81, 'book': 83, 'clock': 84, 'vase': 85, 'scissors': 86, 'teddy bear': 87, 'hair drier': 88, 'toothbrush': 89}

cfg.train_list = ["coco_train.txt"]
cfg.test_list = "coco_val.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.max_epoch = 160
cfg.size_th = 0.1

cfg.gt_from_xml = False
#cfg.annopath = 'VOCdevkit/VOC2007/Annotations/{:s}.xml'
#cfg.imagesetfile = 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'