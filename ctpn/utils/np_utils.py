# -*- coding: utf-8 -*-
"""
   File Name：     np_utils
   Description :  numpy 工具类
   Author :       mick.yi
   date：          2019/2/19
"""

import numpy as np


def pad_to_fixed_size(input_np, fixed_size):
    """
    增加padding到固定尺寸,在第二维增加一个标志位,0-padding,1-非padding
    :param input_np: 二维数组
    :param fixed_size:
    :return:
    """
    shape = input_np.shape
    # 增加tag
    np_array = np.pad(input_np, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    # 增加padding
    pad_num = max(0, fixed_size - shape[0])
    return np.pad(np_array, ((0, pad_num), (0, 0)), mode='constant', constant_values=0)


def remove_pad(input_np):
    """
    去除padding
    :param input_np:
    :return:
    """
    pad_tag = input_np[:, -1]  # 最后一维是padding 标志，1-非padding
    real_size = int(np.sum(pad_tag))
    return input_np[:real_size, :-1]


def compute_iou(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: (N,4)
    :param boxes_b: (M,4)
    :return:  IoU (N,M)
    """
    # 扩维
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1,M,4)

    # 分别计算高度和宽度的交集
    overlap_h = np.maximum(0.0,
                           np.minimum(boxes_a[..., 2], boxes_b[..., 2]) -
                           np.maximum(boxes_a[..., 0], boxes_b[..., 0]))  # (N,M)

    overlap_w = np.maximum(0.0,
                           np.minimum(boxes_a[..., 3], boxes_b[..., 3]) -
                           np.maximum(boxes_a[..., 1], boxes_b[..., 1]))  # (N,M)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    裁剪边框到图像内
    :param boxes: 边框 [n,(y1,x1,y2,x2)]
    :param im_shape: tuple(H,W,C)
    :return:
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1])
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0])
    return boxes
