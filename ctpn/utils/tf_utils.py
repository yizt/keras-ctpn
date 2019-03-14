# -*- coding: utf-8 -*-
"""
   File Name：     tf_utils
   Description :  tensorflow工具类
   Author :       mick.yi
   date：          2019/3/13
"""
import tensorflow as tf
from deprecated import deprecated


@deprecated(reason='建议使用原生tf.map_fn;效率更高,并且不需要显示传入batch_size参数')
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    将输入分片，然后每个分片执行指定计算，最后组合结果;适用于批量处理计算图逻辑只支持一个实例的情况
    :param inputs: tensor列表
    :param graph_fn: 计算逻辑
    :param batch_size:
    :param names:
    :return:
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    # 行转列
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)
    # list转tensor
    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    # 如果返回单个值,不使用list
    if len(result) == 1:
        result = result[0]

    return result


def pad_to_fixed_size_with_negative(input_tensor, fixed_size, negative_num, data_type=tf.float32):
    # 输入尺寸
    input_size = tf.shape(input_tensor)[0]
    # tag 列 padding
    positive_num = input_size - negative_num  # 正例数
    # 正样本padding 1,负样本padding -1
    column_padding = tf.concat([tf.ones([positive_num], data_type),
                                tf.ones([negative_num], data_type) * -1],
                               axis=0)
    # 都转为float,拼接
    x = tf.concat([tf.cast(input_tensor, data_type), tf.expand_dims(column_padding, axis=1)], axis=1)
    # 不够的padding 0
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)
    return x


def pad_to_fixed_size(input_tensor, fixed_size):
    """
    增加padding到固定尺寸,在第二维增加一个标志位,0-padding,1-非padding
    :param input_tensor: 二维张量
    :param fixed_size:
    :param negative_num: 负样本数量
    :return:
    """
    input_size = tf.shape(input_tensor)[0]
    x = tf.pad(input_tensor, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=1)
    # padding
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)
    return x


def pad_list_to_fixed_size(tensor_list, fixed_size):
    return [pad_to_fixed_size(tensor, fixed_size) for tensor in tensor_list]


def remove_pad(input_tensor):
    """

    :param input_tensor:
    :return:
    """
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


def clip_boxes(boxes, window):
    """
    将boxes裁剪到指定的窗口范围内
    :param boxes: 边框坐标，[N,(y1,x1,y2,x2)]
    :param window: 窗口坐标，[(y1,x1,y2,x2)]
    :return:
    """
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)  # split后维数不变

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)  # wy1<=y1<=wy2
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

    clipped_boxes = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    # clipped_boxes.([boxes.shape[0], 4])
    return clipped_boxes


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)
    w *= tf.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return tf.stack([y1, x1, y2, x2], axis=1)
