# -*- coding: utf-8 -*-
"""
   File Name：     gt_utils
   Description :
   Author :       mick.yi
   date：          2019/3/18
"""
import numpy as np


def linear_fit_y(xs, ys, x_list):
    """
    线性函数拟合两点(x1,y1),(x2,y2)；并求得x_set在的取值
    :param xs:  [x1,x2]
    :param ys:  [y1,y2]
    :param x_list: x轴坐标点,numpy数组 [n]
    :return:
    """
    if xs[0] == xs[1]:  # 垂直线
        return np.ones_like(x_list) * np.mean(ys)
    elif ys[0] == ys[1]:  # 水平线
        return np.ones_like(x_list) * ys[0]
    else:
        fn = np.poly1d(np.polyfit(xs, ys, 1))  # 一元线性函数
        return fn(x_list)


def get_min_max_y(quadrilateral, xs):
    """
    获取指定x值坐标点集合四边形上的y轴最小值和最大值
    :param quadrilateral: 四边形坐标；x1,y1,x2,y2,x3,y3,x4,y4
    :param xs: x轴坐标点,numpy数组 [n]
    :return:  x轴坐标点在四边形上的最小值和最大值
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quadrilateral.tolist()
    y_val_1 = linear_fit_y(np.array([x1, x2]), np.array([y1, y2]), xs)
    y_val_2 = linear_fit_y(np.array([x2, x3]), np.array([y2, y3]), xs)
    y_val_3 = linear_fit_y(np.array([x3, x4]), np.array([y3, y4]), xs)
    y_val_4 = linear_fit_y(np.array([x4, x1]), np.array([y4, y1]), xs)
    y_val = np.stack([y_val_1, y_val_2, y_val_3, y_val_4], axis=1)
    y_val_sort = np.sort(y_val, axis=1)
    return y_val_sort[:, 1], y_val_sort[:, 2]


def get_xs_in_range(x_array, x_min, x_max):
    indices = np.logical_and(x_array >= x_min, x_array <= x_max)
    xs = x_array[indices]
    # 处理两端的值
    if xs.shape[0] == 0 or xs[0] > x_min:
        xs = np.insert(xs, 0, x_min)
    if xs.shape[0] == 0 or xs[-1] < x_max:
        xs = np.append(xs, x_max)

    return xs


def gen_gt_from_quadrilaterals(gt_quadrilaterals, input_gt_class_ids, image_shape, width_stride, box_min_height=8):
    h, w = list(image_shape)[:2]
    x_array = np.arange(0, w + 1, width_stride, np.float32)  # 固定宽度间隔的x坐标点
    # 每个四边形x 最小值和最大值
    x_min_np = np.min(gt_quadrilaterals[:, ::2], axis=1)
    x_max_np = np.max(gt_quadrilaterals[:, ::2], axis=1)
    gt_boxes = []
    gt_class_ids = []
    for i in np.arange(len(gt_quadrilaterals)):
        xs = get_xs_in_range(x_array, x_min_np[i], x_max_np[i])  # 获取四边形内的x中坐标点
        ys_min, ys_max = get_min_max_y(gt_quadrilaterals[i], xs)
        # 为每个四边形生成固定宽度的gt
        for j in range(len(xs) - 1):
            x1, x2 = xs[j], xs[j + 1]
            y1, y2 = np.min(ys_min[j:j + 2]), np.max(ys_max[j:j + 2])
            gt_boxes.append([y1, x1, y2, x2])
            gt_class_ids.append(input_gt_class_ids[i])
    gt_boxes = np.reshape(np.array(gt_boxes), (-1, 4))
    gt_class_ids = np.reshape(np.array(gt_class_ids), (-1,))
    # 过滤高度太小的边框
    height = gt_boxes[:, 2] - gt_boxes[:, 0]
    indices = np.where(height >= box_min_height)
    return gt_boxes[indices], gt_class_ids[indices]
