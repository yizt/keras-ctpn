# -*- coding: utf-8 -*-
"""
   File Name：     Target
   Description :   分类和回归目标层
   Author :       mick.yi
   date：          2019/3/13
"""

from keras import layers
import tensorflow as tf
from ..utils import tf_utils


def compute_iou(gt_boxes, anchors):
    """
    计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    return iou


def ctpn_regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N, (dy, dh)]
    """
    # anchor高度
    h = anchors[:, 2] - anchors[:, 0]
    # gt高度
    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]

    # anchor中心点y坐标
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    # gt中心点y坐标
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5

    # 计算回归目标
    dy = (gt_center_y - center_y) / h
    dh = tf.log(gt_h / h)

    target = tf.stack([dy, dh], axis=1)
    target /= tf.constant([0.1, 0.2])

    return target


def side_regress_target(anchors, gt_boxes, side='left'):
    """

    :param anchors:
    :param gt_boxes:
    :param side:
    :return:
    """
    w = anchors[:, 3] - anchors[:, 1]
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    # 判断是左侧还是右侧
    if side == 'left':
        target = (gt_boxes[:, 1] - center_x) / w
    else:
        target = (gt_boxes[:, 3] - center_x) / w

    return target / 0.1


def ctpn_target_graph(gt_boxes, gt_cls, anchors, valid_anchors_indices, train_anchors_num=128, positive_ratios=0.5,
                      max_gt_num=50):
    """
    处理单个图像的ctpn回归目标
    a)正样本: 与gt IoU大于0.7的anchor,或者与GT IoU最大的那个anchor
    b)需要保证所有的GT都有anchor对应
    :param gt_boxes: gt边框坐标 [gt_num, (y1,x1,y2,x2,tag)], tag=0为padding
    :param gt_cls: gt类别 [gt_num, 1+1], 最后一位为tag, tag=0为padding
    :param anchors: [anchor_num, (y1,x1,y2,x2)]
    :param valid_anchors_indices:有效的anchors索引 [anchor_num]
    :param train_anchors_num
    :param positive_ratios
    :param max_gt_num
    :return:
    deltas:[train_anchors_num, (dy,dh,tag)],anchor边框回归目标,tag=1为正样本,tag=0为padding
    class_id:[train_anchors_num,(class_id,tag)]
    indices: [train_anchors_num,(anchors_index,tag)] tag=1为正样本,tag=0为padding,-1为负样本
    side_deltas_all：[max_gt_num,(left_deltas,right_deltas,left_index,right_index,tag)]
    """
    # 获取真正的GT,去除标签位
    gt_boxes = tf_utils.remove_pad(gt_boxes)
    gt_cls = tf_utils.remove_pad(gt_cls)[:, 0]  # [N,1]转[N]

    gt_num = tf.shape(gt_cls)[0]  # gt 个数

    # 计算IoU
    iou = compute_iou(gt_boxes, anchors)
    # 每个GT对应的IoU最大的anchor是正样本(一般有多个)
    gt_iou_max = tf.reduce_max(iou, axis=1, keep_dims=True)  # 每个gt最大的iou [gt_num,1]
    gt_iou_max_bool = tf.equal(iou, gt_iou_max)  # bool类型[gt_num,num_anchors];每个gt最大的iou(可能多个)

    # 每个anchors最大iou ，且iou>0.7的为正样本
    anchors_iou_max = tf.reduce_max(iou, axis=0, keep_dims=True)  # 每个anchor最大的iou; [1,num_anchors]
    anchors_iou_max = tf.where(tf.greater_equal(anchors_iou_max, 0.7),
                               anchors_iou_max,
                               tf.ones_like(anchors_iou_max))
    anchors_iou_max_bool = tf.equal(iou, anchors_iou_max)

    # 合并两部分正样本索引
    positive_bool_matrix = tf.logical_or(gt_iou_max_bool, anchors_iou_max_bool)
    positive_indices = tf.where(positive_bool_matrix)
    # 采样正样本
    positive_num = tf.minimum(tf.shape(positive_indices)[0], int(train_anchors_num * positive_ratios))
    positive_indices = tf.random_shuffle(positive_indices)[:positive_num]

    # 获取正样本和对应的GT
    positive_gt_indices = positive_indices[:, 0]
    positive_anchor_indices = positive_indices[:, 1]
    positive_anchors = tf.gather(anchors, positive_anchor_indices)
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_indices)
    positive_gt_cls = tf.gather(gt_cls, positive_gt_indices)

    # 计算回归目标
    deltas = ctpn_regress_target(positive_anchors, positive_gt_boxes)

    # 找到每个gt两端的anchor
    anchors_x1 = anchors[:, 1]  # anchors的左侧坐标
    anchors_x1 = tf.tile(tf.expand_dims(anchors_x1, axis=0), [gt_num, 1])  # 扩充为二维[gt_num,num_anchors]

    # gt左右两侧对应的anchor;每个GT对应左右两侧
    gt_left_anchors_index = tf.argmin(
        tf.where(positive_bool_matrix, anchors_x1, tf.ones_like(anchors_x1) * 10.0 ** 10), axis=1)
    gt_right_anchors_index = tf.argmax(
        tf.where(positive_bool_matrix, anchors_x1, tf.ones_like(anchors_x1) * -10.0 ** 10), axis=1)

    gt_left_anchors = tf.gather(anchors, gt_left_anchors_index)
    gt_right_anchors = tf.gather(anchors, gt_right_anchors_index)

    # 侧边回归目标
    side_left_deltas = side_regress_target(gt_left_anchors, gt_boxes, 'left')
    side_right_deltas = side_regress_target(gt_right_anchors, gt_boxes, 'right')

    side_deltas = tf.stack([side_left_deltas, side_right_deltas], axis=1)
    # 对应到有效的anchors
    side_indices = tf.stack([tf.gather(valid_anchors_indices, gt_left_anchors_index),
                             tf.gather(valid_anchors_indices, gt_right_anchors_index)], axis=1)

    # # 获取负样本 iou<0.5
    negative_bool = tf.less(tf.reduce_max(iou, axis=0), 0.5)
    positive_bool = tf.reduce_any(positive_bool_matrix, axis=0)  # 正样本anchors [num_anchors]
    negative_bool = tf.logical_and(negative_bool, tf.logical_not(positive_bool))

    # 采样负样本
    negative_num = tf.minimum(int(train_anchors_num * (1. - positive_ratios)), train_anchors_num - positive_num)
    negative_indices = tf.random_shuffle(tf.where(negative_bool)[:, 0])[:negative_num]

    negative_gt_cls = tf.zeros([negative_num])  # 负样本类别id为0
    negative_deltas = tf.zeros([negative_num, 2])

    # 合并正负样本
    deltas = tf.concat([deltas, negative_deltas], axis=0, name='ctpn_target_deltas')
    class_ids = tf.concat([positive_gt_cls, negative_gt_cls], axis=0, name='ctpn_target_class_ids')
    indices = tf.concat([tf.gather(valid_anchors_indices, positive_anchor_indices), negative_indices], axis=0,
                        name='ctpn_train_anchor_indices')

    # 计算padding
    deltas, class_ids = tf_utils.pad_list_to_fixed_size([deltas, tf.expand_dims(class_ids, 1)],
                                                        train_anchors_num)
    # 将负样本tag标志改为-1;方便后续处理;
    indices = tf_utils.pad_to_fixed_size_with_negative(tf.expand_dims(indices, 1), train_anchors_num,
                                                       negative_num=negative_num, data_type=tf.int64)

    side_deltas, side_indices = tf_utils.pad_list_to_fixed_size([side_deltas, side_indices], max_gt_num)

    return [deltas, class_ids, indices, side_deltas, side_indices, tf.cast(  # 用作度量的必须是浮点类型
        gt_num, dtype=tf.float32), tf.cast(
        positive_num, dtype=tf.float32), tf.cast(negative_num, dtype=tf.float32)]


def get_real_anchors_indices(indices, valid_anchors_indices):
    """

    :param indices: [N]
    :param valid_anchors_indices: [M]
    :return:
    """
    return tf.gather(valid_anchors_indices, indices)


class CtpnTarget(layers.Layer):
    def __init__(self, batch_size, train_anchors_num=128, positive_ratios=0.5, max_gt_num=50, **kwargs):
        self.batch_size = batch_size
        self.train_anchors_num = train_anchors_num
        self.positive_ratios = positive_ratios
        self.max_gt_num = max_gt_num
        super(CtpnTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        inputs[0]: GT 边框坐标 [batch_size, MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=0 为padding
        inputs[1]: GT 类别 [batch_size, MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=0 为padding
        inputs[2]: Anchors [batch_size, anchor_num,(y1,x1,y2,x2)]
        inputs[3]: val_anchors_indices [batch_size, anchor_num]
        :param kwargs:
        :return:
        """
        gt_boxes, gt_cls_ids, anchors, valid_anchors_indices = inputs
        # options = {"train_anchors_num": self.train_anchors_num,
        #            "positive_ratios": self.positive_ratios,
        #            "max_gt_num": self.max_gt_num}
        #
        # outputs = tf.map_fn(fn=lambda x: ctpn_target_graph(*x, **options),
        #                     elems=[gt_boxes, gt_cls_ids, anchors, valid_anchors_indices],
        #                     dtype=[tf.float32] * 2 + [tf.int64] + [tf.float32] + [tf.int64] + [tf.float32] * 3)
        outputs = tf_utils.batch_slice([gt_boxes, gt_cls_ids, anchors, valid_anchors_indices],
                                       lambda x, y, z, s: ctpn_target_graph(x, y, z, s,
                                                                            self.train_anchors_num,
                                                                            self.positive_ratios,
                                                                            self.max_gt_num),
                                       batch_size=self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.train_anchors_num, 3),  # deltas (dy,dh)
                (input_shape[0][0], self.train_anchors_num, 2),  # cls
                (input_shape[0][0], self.train_anchors_num, 2),  # indices
                (input_shape[0][0], self.max_gt_num, 3),  # side_deltas
                (input_shape[0][0], self.max_gt_num, 3),  # side_indices
                (input_shape[0][0],),  # gt_num
                (input_shape[0][0],),  # positive_num
                (input_shape[0][0],)]  # negative_num
