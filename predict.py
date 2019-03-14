# -*- coding: utf-8 -*-
"""
   File Name：     predict
   Description :   模型预测
   Author :       mick.yi
   date：          2019/3/14
"""
import sys
import numpy as np
from .utils import image_utils, np_utils
from .utils.detector import TextDetector
from .config import cur_config as config
from .layers import models


def main(image_path):
    # 加载图片
    image, image_meta, _ = image_utils.load_image_gt(id,
                                                     image_path,
                                                     config.IMAGE_SHAPE[0],
                                                     None)
    # 加载模型
    config.IMAGES_PER_GPU = 1
    m = models.ctpn_net(config, 'test')
    m.load_weights(config.WEIGHT_PATH)
    # m.summary()

    # 模型预测
    text_boxes, text_scores, _ = m.predict(np.array(image))
    text_boxes = np_utils.remove_pad(text_boxes[0])
    text_scores = np_utils.remove_pad(text_scores[0])
    # 文本行检测器
    detector = TextDetector(config)
    text_lines = detector.detect(text_boxes, text_scores, config.IMAGE_SHAPE)
    print("text_lines:{}".format(text_lines))


if __name__ == '__main__':
    main(sys.argv[1])
