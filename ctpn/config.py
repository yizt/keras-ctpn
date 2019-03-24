# -*- coding: utf-8 -*-
"""
   File Name：     config
   Description :  配置类
   Author :       mick.yi
   date：          2019/3/14
"""


class Config(object):
    IMAGES_PER_GPU = 4
    IMAGE_SHAPE = (720, 720, 3)
    MAX_GT_INSTANCES = 1000

    NUM_CLASSES = 1 + 1  #
    CLASS_MAPPING = {'bg': 0,
                     'text': 1}
    # 训练样本
    ANCHORS_HEIGHT = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    ANCHORS_WIDTH = 16
    TRAIN_ANCHORS_PER_IMAGE = 128
    ANCHOR_POSITIVE_RATIO = 0.5
    # 步长
    NET_STRIDE = 16
    # text proposal输出
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    TEXT_PROPOSALS_MAX_NUM = 500
    TEXT_PROPOSALS_WIDTH = 16
    # text line boxes超参数
    LINE_MIN_SCORE = 0.7
    MAX_HORIZONTAL_GAP = 50
    TEXT_LINE_NMS_THRESH = 0.3
    MIN_NUM_PROPOSALS = 1
    MIN_RATIO = 1.2
    MIN_V_OVERLAPS = 0.7
    MIN_SIZE_SIM = 0.7

    # 训练超参数
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9
    # 权重衰减
    WEIGHT_DECAY = 0.0005,
    GRADIENT_CLIP_NORM = 5.0

    LOSS_WEIGHTS = {
        "ctpn_regress_loss": 1.,
        "ctpn_class_loss": 1,
        "side_regress_loss": 1
    }
    # 是否使用侧边改善
    USE_SIDE_REFINE = True
    # 预训练模型
    PRE_TRAINED_WEIGHT = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    WEIGHT_PATH = '/tmp/ctpn.h5'

    # 数据集路径
    IMAGE_DIR = '/opt/dataset/OCR/ICDAR_2015/train_images'
    IMAGE_GT_DIR = '/opt/dataset/OCR/ICDAR_2015/train_gt'


cur_config = Config()
