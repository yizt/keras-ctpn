# -*- coding: utf-8 -*-
"""
   File Name：     predict
   Description :   模型预测
   Author :       mick.yi
   date：          2019/3/14
"""
import sys
import numpy as np
import argparse
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ctpn.utils import image_utils, np_utils, visualize
from ctpn.utils.detector import TextDetector
from ctpn.config import cur_config as config
from ctpn.layers import models


def main(args):
    # 加载图片
    image, image_meta, _ = image_utils.load_image_gt(id,
                                                     args.image_path,
                                                     config.IMAGE_SHAPE[0],
                                                     None)
    # 加载模型
    config.IMAGES_PER_GPU = 1
    m = models.ctpn_net(config, 'test')
    if args.weight_path is not None:
        m.load_weights(args.weight_path)
    else:
        m.load_weights(config.WEIGHT_PATH)
    # m.summary()

    # 模型预测
    text_boxes, text_scores, _ = m.predict(np.array([image]))
    text_boxes = np_utils.remove_pad(text_boxes[0])
    text_scores = np_utils.remove_pad(text_scores[0])[:, 0]
    # print("text_scores:{}".format(text_scores))
    # print("text_boxes:{}".format(text_boxes))
    # 文本行检测器
    detector = TextDetector(config)
    text_lines = detector.detect(text_boxes, text_scores, config.IMAGE_SHAPE)
    print("text_lines:{}".format(text_lines))

    boxes_num = 3
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    visualize.display_instances(image, text_lines[:boxes_num, :4],
                                np.ones_like(text_lines[:boxes_num, 4], np.int32),
                                ['bg', 'text'],
                                scores=text_lines[:boxes_num, 4],
                                ax=ax)
    fig.savefig('examples.{}.png'.format(np.random.randint(10)))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, help="image path")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
