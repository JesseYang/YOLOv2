from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 416
cfg.img_h = 416
cfg.grid_w = 32
cfg.grid_h = 32

cfg.multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448], [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]]

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

'''
train
Number of images: 66843
k-means clustering pascal anchor points (original coordinates)
Found at iteration 4 with best average IoU: 0.5163343295040421 
[[ 0.4930636   0.83190876]
 [ 1.34735609  2.61651251]
 [ 2.83276422  6.79906007]
 [ 5.08603695  3.11557501]
 [ 8.82067671  9.3150539 ]]

trainval
k-means clustering pascal anchor points (original coordinates)
Found at iteration 4 with best average IoU: 0.516533648800776 
[[ 0.49754394  0.83553621]
 [ 1.35252889  2.62245182]
 [ 2.84472443  6.79466355]
 [ 5.10358836  3.1065874 ]
 [ 8.83700452  9.33602367]]

voc07 voc12 coco
 k-means clustering pascal anchor points (original coordinates)
Found at iteration 56 with best average IoU: 0.5153392451551813 
[[  0.52340679   0.89816029]
 [  1.62204465   2.80834417]
 [  3.12342422   7.01347654]
 [  7.18121243   4.15297634]
 [  9.25215845  10.1937983 ]]

'''

cfg.anchors = [[  0.52340679,   0.89816029],
               [  1.62204465,   2.80834417],
               [  3.12342422,   7.01347654],
               [  7.18121243,   4.15297634],
               [  9.25215845,  10.1937983 ]]

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
	               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
	               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
	               'sofa': 17, 'train': 18, 'tvmonitor': 19}

cfg.train_list = ["coco_voc_train.txt", "coco_voc_val.txt", "voc_2007_train.txt", "voc_2012_train.txt", "voc_2007_val.txt", "voc_2012_val.txt"]
cfg.test_list = "voc_2007_test_without_diff.txt"

#cfg.train_list = ["yolo_code_test2.txt"]
#cfg.test_list = "yolo_code_test2.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.max_epoch = 160
cfg.size_th = 0.1

cfg.gt_from_xml = False
#cfg.gt_from_xml = True
#cfg.annopath = 'VOCdevkit/VOC2007/Annotations/{:s}.xml'
#cfg.imagesetfile = 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'