from .config_coco_voc import cfg
# from .config_kitti_3cls import cfg
# from .config_crop_person import cfg

cfg.lr_schedule = [(0, 1e-4), (3, 2e-4), (6, 3e-4), (10, 6e-4), (15, 1e-3), (60, 1e-4), (90, 1e-5)]
