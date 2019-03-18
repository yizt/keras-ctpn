# -*- coding: utf-8 -*-
"""
   File Name：     gt
   Description :  根据GT生成宽度为16的系列GT
   Author :       mick.yi
   date：          2019/3/18
"""
import keras
import tensorflow as tf
from ..utils import tf_utils, gt_utils
from deprecated import deprecated


def generate_gt_graph(gt_quadrilaterals, input_gt_class_ids, image_shape, width_stride, max_gt_num):
    """

    :param gt_quadrilaterals: [mat_gt_num,(x1,y1,x2,y2,x3,y3,x4,y4,tag)] 左上、右上、右下、左下(顺时针)
    :param input_gt_class_ids:
    :param image_shape:
    :param width_stride:
    :param max_gt_num:
    :return:
    """

    gt_quadrilaterals = tf_utils.remove_pad(gt_quadrilaterals)
    input_gt_class_ids = tf_utils.remove_pad(input_gt_class_ids)
    gt_boxes, gt_class_ids = tf.py_func(func=gt_utils.gen_gt_from_quadrilaterals,
                          inp=[gt_quadrilaterals, input_gt_class_ids, image_shape, width_stride],
                          Tout=[tf.float32] * 2)
    return tf_utils.pad_list_to_fixed_size([gt_boxes, gt_class_ids], max_gt_num)


@deprecated(reason='目前没有用')
class GenGT(keras.layers.Layer):
    def __init__(self, image_shape, width_stride, max_gt_num, **kwargs):
        self.image_shape = image_shape
        self.width_stride = width_stride
        self.max_gt_num = max_gt_num
        super(GenGT, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs: gt_quadrilaterals [batch_size,mat_gt_num,(y1,x1,y2,x1,tag)]
        :param kwargs:
        :return:
        """
        gt_quadrilaterals = inputs[0]
        input_gt_class_ids = inputs[1]
        outputs = tf_utils.batch_slice([gt_quadrilaterals, input_gt_class_ids],
                                       lambda x: generate_gt_graph(x, self.image_shape,
                                                                   self.width_stride, self.max_gt_num),
                                       batch_size=self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.max_gt_num, 5),
                (input_shape[0][0], self.max_gt_num, 2)]
