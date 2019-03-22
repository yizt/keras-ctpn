# -*- coding: utf-8 -*-
"""
   File Name：     text_proposals
   Description :  文本提议框生成
   Author :       mick.yi
   date：          2019/3/14
"""
from keras import layers
import tensorflow as tf
from ..utils import tf_utils


def apply_regress(deltas, side_deltas, anchors, use_side_refine=False):
    """
    应用回归目标到边框, 垂直中心点偏移和高度缩放
    :param deltas: 回归目标[N,(dy,dh,)]
    :param side_deltas: 回归目标[N,(dx)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :param use_side_refine: 是否应用侧边回归
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    deltas = tf.concat([deltas, side_deltas], axis=1)
    # 回归系数
    deltas *= tf.constant([0.1, 0.2, 0.1])
    dy, dh, dx = deltas[:, 0], deltas[:, 1], deltas[:, 2]

    # 中心坐标回归
    cy += dy * h
    # 侧边精调
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    y2 = cy + h * 0.5
    x1 = tf.maximum(cx - w * 0.5, 0.)  # 限制在窗口内,修复后继节点找不到对应的前驱节点
    x2 = cx + w * 0.5

    if use_side_refine:
        return tf.stack([y1, x1, y2, x2], axis=1)
    else:
        return tf.stack([y1, anchors[:, 1], y2, anchors[:, 3]], axis=1)


def get_valid_predicts(deltas, side_deltas, class_logits, valid_anchors_indices):
    return tf.gather(deltas, valid_anchors_indices), tf.gather(
        side_deltas, valid_anchors_indices), tf.gather(
        class_logits, valid_anchors_indices)


def nms(boxes, scores, class_logits, max_output_size, iou_threshold=0.5, score_threshold=0.05,
        name=None):
    """
    非极大抑制
    :param boxes: 形状为[num_boxes, 4]的二维浮点型Tensor.
    :param scores: 形状为[num_boxes]的一维浮点型Tensor,表示与每个框(每行框)对应的单个分数.
    :param class_logits: 形状为[num_boxes,num_classes] 原始的预测类别
    :param max_output_size: 一个标量整数Tensor,表示通过非最大抑制选择的框的最大数量.
    :param iou_threshold: 浮点数,IOU 阈值
    :param score_threshold:  浮点数, 过滤低于阈值的边框
    :param name:
    :return: 检测边框、边框得分、边框类别
    """
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold,
                                           name)  # 一维索引
    output_boxes = tf.gather(boxes, indices)  # (M,4)
    class_scores = tf.expand_dims(tf.gather(scores, indices), axis=1)  # 扩展到二维(M,1)
    class_logits = tf.gather(class_logits, indices)
    # padding到固定大小
    return [tf_utils.pad_to_fixed_size(output_boxes, max_output_size),
            tf_utils.pad_to_fixed_size(class_scores, max_output_size),
            tf_utils.pad_to_fixed_size(class_logits, max_output_size)]


class TextProposal(layers.Layer):
    """
    生成候选框
    """

    def __init__(self, batch_size, score_threshold=0.7, output_box_num=500, iou_threshold=0.3,
                 use_side_refine=False, **kwargs):
        """

        :param score_threshold: 分数阈值
        :param output_box_num: 生成proposal 边框数量
        :param iou_threshold: nms iou阈值
        :param use_side_refine : 预测时是否使用侧边改善
        """
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.output_box_num = output_box_num
        self.iou_threshold = iou_threshold
        self.use_side_refine = use_side_refine
        super(TextProposal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [batch_size,N,(dy,dh)]   N是所有的anchors数量
        inputs[1]: side_deltas, [batch_size,N,(dx)]   N是所有的anchors数量
        inputs[2]: class logits [batch_size,N,num_classes]
        inputs[3]: anchors [batch_size,N,(y1,x1,y2,x2)]
        inputs[4]: valid_anchors_indices [batch_size, anchor_num]
        :param kwargs:
        :return:
        """
        deltas, side_deltas, class_logits, anchors, val_anchors_indices = inputs
        # 只看有效anchor的预测结果
        deltas, side_deltas, class_logits = tf_utils.batch_slice(
            [deltas, side_deltas, class_logits, val_anchors_indices],
            lambda x, y, z, u: get_valid_predicts(x, y, z, u),
            self.batch_size)
        # 转为分类评分
        class_scores = tf.nn.softmax(logits=class_logits, axis=-1)  # [N,num_classes]
        fg_scores = tf.reduce_max(class_scores[..., 1:], axis=-1)  # 第一类为背景 (N,)

        # # 应用边框回归
        # proposals = tf.map_fn(fn=lambda x: apply_regress(*x),
        #                       elems=[deltas, anchors],
        #                       dtype=tf.float32)
        # # # 非极大抑制
        #
        # options = {"max_output_size": self.output_box_num,
        #            "iou_threshold": self.iou_threshold,
        #            "score_threshold": self.score_threshold}
        # outputs = tf.map_fn(fn=lambda x: nms(*x, **options),
        #                     elems=[proposals, fg_scores, class_logits],
        #                     dtype=[tf.float32] * 3)
        proposals = tf_utils.batch_slice([deltas, side_deltas, anchors],
                                         lambda x, y, z: apply_regress(x, y, z, self.use_side_refine),
                                         self.batch_size)

        outputs = tf_utils.batch_slice([proposals, fg_scores, class_logits],
                                       lambda x, y, z: nms(x, y, z,
                                                           max_output_size=self.output_box_num,
                                                           iou_threshold=self.iou_threshold,
                                                           score_threshold=self.score_threshold),
                                       self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        """
        注意多输出，call返回值必须是列表
        :param input_shape:
        :return:
        """
        return [(input_shape[0][0], self.output_box_num, 4 + 1),
                (input_shape[0][0], self.output_box_num, 1 + 1),
                (input_shape[0][0], self.output_box_num, input_shape[1][-1])]
