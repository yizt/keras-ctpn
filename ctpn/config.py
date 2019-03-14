# -*- coding: utf-8 -*-
"""
   File Name：     config
   Description :
   Author :       mick.yi
   date：          2019/3/14
"""


class Config(object):
    NUM_CLASSES = 1 + 1  #
    CLASS_MAPPING = {'bg': 0,
                     'text': 1}
    # 训练超参数
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001,
    GRADIENT_CLIP_NORM = 5.0

    LOSS_WEIGHTS = {
        "ctpn_regress_loss": 1.,
        "ctpn_class_loss": 1
    }
