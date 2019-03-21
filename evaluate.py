# -*- coding: utf-8 -*-
"""
   File Name：     evaluate
   Description :  评估入口
   Author :       mick.yi
   date：          2019/3/21
"""

import sys
import os
import numpy as np
import argparse
from ctpn.utils import image_utils, file_utils, np_utils
from ctpn.utils.detector import TextDetector
from ctpn.config import cur_config as config
from ctpn.layers import models
import datetime


def generator(image_path_list, image_shape):
    for i, image_path in enumerate(image_path_list):
        image, image_meta, _, _ = image_utils.load_image_gt(np.random.randint(10),
                                                            image_path,
                                                            image_shape[0])
        if i % 200 == 0:
            print("开始评估第 {} 张图像".format(i))
        yield {"input_image": np.asarray([image]),
               "input_image_meta": np.asarray([image_meta])}


def main(args):
    # 覆盖参数
    config.USE_SIDE_REFINE = bool(args.use_side_refine)
    if args.weight_path is not None:
        config.WEIGHT_PATH = args.weight_path
    config.IMAGES_PER_GPU = 1
    # 图像路径
    image_path_list = file_utils.get_sub_files(args.image_dir)

    # 加载模型
    m = models.ctpn_net(config, 'test')
    m.load_weights(config.WEIGHT_PATH, by_name=True)

    # 预测
    start_time = datetime.datetime.now()
    gen = generator(image_path_list, config.IMAGE_SHAPE)
    text_boxes, text_scores, image_metas = m.predict_generator(generator=gen,
                                                               steps=len(image_path_list),
                                                               use_multiprocessing=True)
    end_time = datetime.datetime.now()
    print("======完成{}张图像评估，耗时:{} 秒".format(len(image_path_list), end_time-start_time))
    # 去除padding
    text_boxes = [np_utils.remove_pad(text_box) for text_box in text_boxes]
    text_scores = [np_utils.remove_pad(text_score)[:, 0] for text_score in text_scores]
    image_metas = image_utils.batch_parse_image_meta(image_metas)
    # 文本行检测
    detector = TextDetector(config)
    text_lines = [detector.detect(boxes, scores, config.IMAGE_SHAPE, window)
                  for boxes, scores, window in zip(text_boxes, text_scores, image_metas["window"])]
    # 还原检测文本行边框到原始图像坐标
    text_lines = [image_utils.recover_detect_boxes(boxes, window, scale)
                  for boxes, window, scale in zip(text_lines, image_metas["window"], image_metas["scale"])]

    # 写入文档中
    for image_path, boxes in zip(image_path_list, text_lines):
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        with open(os.path.join(args.output_dir, output_filename), mode='w') as f:
            for box in boxes:
                f.write("{},{},{},{},{},{},{},{}\r\n".format(box[0],
                                                             box[1],
                                                             box[2],
                                                             box[3],
                                                             box[4],
                                                             box[5],
                                                             box[6],
                                                             box[7]))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_dir", type=str, help="image dir")
    parse.add_argument("--output_dir", type=str, help="output dir")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--use_side_refine", type=int, default=1, help="1: use side refine; 0 not use")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
