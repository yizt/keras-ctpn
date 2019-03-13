# -*- coding: utf-8 -*-
"""
   File Name：     losses
   Description :  损失函数层
   Author :       mick.yi
   date：          2019/3/13
"""
import tensorflow as tf
from keras import backend as K


def ctpn_cls_loss(predict_cls_ids, true_cls_ids, indices):
    """
    ctpn分类损失
    :param predict_cls_ids: 预测的anchors类别，(batch_num,anchors_num,2) fg or bg
    :param true_cls_ids:实际的anchors类别，(batch_num,rpn_train_anchors,(class_id,tag))
             tag 1：正样本，0：负样本，-1 padding
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,(idx,tag))，
             idx:指定anchor索引位置，tag 1：正样本，0：负样本，-1 padding
    :return:
    """
    # 去除padding
    train_indices = tf.where(tf.not_equal(indices[:, :, -1], 0))  # 0为padding
    train_anchor_indices = tf.gather_nd(indices[..., 0], train_indices)  # 一维(batch*train_num,)，每个训练anchor的索引
    true_cls_ids = tf.gather_nd(true_cls_ids[..., 0], train_indices)  # 一维(batch*train_num,)
    # 转为onehot编码
    true_cls_ids = tf.where(true_cls_ids >= 1,
                            tf.ones_like(true_cls_ids, dtype=tf.uint8),
                            tf.zeros_like(true_cls_ids, dtype=tf.uint8))  # 前景类都为1
    true_cls_ids = tf.one_hot(true_cls_ids, depth=2)
    # batch索引
    batch_indices = train_indices[:, 0]  # 训练的第一维是batch索引
    # 每个训练anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(train_anchor_indices, dtype=tf.int64)], axis=1)
    # 获取预测的anchors类别
    predict_cls_ids = tf.gather_nd(predict_cls_ids, train_indices_2d)  # (batch*train_num,2)

    # 交叉熵损失函数
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_cls_ids, logits=predict_cls_ids)
    return losses


def smooth_l1_loss(y_true, y_predict, sigma2=9.0):
    """
    smooth L1损失函数；   0.5 * sigma2 * x^2 if |x| <1/sigma2 else |x|-0.5/sigma2; x是 diff
    :param y_true:[N,4]
    :param y_predict:[N,4]
    :param sigma2
    :return:
    """
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1./sigma2), 0.5 * sigma2 * tf.pow(abs_diff, 2), abs_diff - 0.5/sigma2)
    return tf.reduce_mean(loss, axis=1)


def ctpn_regress_loss(predict_deltas, deltas, indices):
    """

    :param predict_deltas: 预测的回归目标，(batch_num, anchors_num, 2)
    :param deltas: 真实的回归目标，(batch_num, ctpn_train_anchors, 2+1), 最后一位为tag, tag=0 为padding
    :param indices: 正负样本索引，(batch_num, ctpn_train_anchors, (idx,tag))，
             idx:指定anchor索引位置，最后一位为tag, tag=0 为padding; 1为正样本，-1为负样本
    :return:
    """
    # 去除padding和负样本
    positive_indices = tf.where(tf.equal(indices[:, :, -1], 1))
    deltas = tf.gather_nd(deltas[..., :-1], positive_indices)  # (n,(dy,dh))
    true_positive_indices = tf.gather_nd(indices[..., 0], positive_indices)  # 一维，正anchor索引

    # batch索引
    batch_indices = positive_indices[:, 0]
    # 正样本anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(true_positive_indices, dtype=tf.int64)], axis=1)
    # 正样本anchor预测的回归类型
    predict_deltas = tf.gather_nd(predict_deltas, train_indices_2d, name='ctpn_regress_loss_predict_deltas')

    # Smooth-L1 # 非常重要，不然报NAN
    loss = K.switch(tf.size(deltas) > 0,
                    smooth_l1_loss(deltas, predict_deltas),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

