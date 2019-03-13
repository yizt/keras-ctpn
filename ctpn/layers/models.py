# -*- coding: utf-8 -*-
"""
   File Name：     models
   Description :  模型
   Author :       mick.yi
   date：          2019/3/13
"""
from keras import layers
from keras import Input,Model
from .base_net import resnet50
from .anchor import CtpnAnchor
from .target import CtpnTarget
from .losses import ctpn_cls_loss,ctpn_regress_loss


def ctpn_net(batch_size, image_shape, heights, strides, width, max_gt_num, stage='train'):
    # 网络构建
    input_image = Input(batch_shape=(batch_size,) + image_shape, name='input_image')
    input_image_meta = Input(batch_shape=(batch_size, 12), name='input_image_meta')
    gt_class_ids = Input(batch_shape=(batch_size, max_gt_num, 2), name='gt_class_ids')
    gt_boxes = Input(batch_shape=(batch_size, max_gt_num, 5), name='gt_boxes')
    # 预测
    base_features = resnet50(input_image)
    num_anchors = len(heights)
    predict_class_logits, predict_deltas, predict_side_deltas = ctpn(base_features, num_anchors)

    # anchors生成
    anchors = CtpnAnchor(heights, width, strides, image_shape, name='gen_ctpn_anchors')

    if stage == 'train':
        targets = CtpnTarget(name='ctpn_target')(gt_boxes,gt_class_ids,anchors)
        deltas,class_ids,anchors_indices,side_deltas=targets[:4]
        # 损失函数
        regress_loss=layers.Lambda(lambda x:ctpn_regress_loss(*x),
                                   name='ctpn_regress_loss')([predict_deltas,deltas,anchors_indices])
        cls_loss=layers.Lambda(lambda x:ctpn_cls_loss(*x),
                               name='ctpn_class_loss')([predict_class_logits,class_ids,anchors_indices])
        model=Model(inputs=[input_image,gt_boxes,gt_class_ids],outputs=[regress_loss,cls_loss])

    else:
        pass


def ctpn(base_features, num_anchors, rnn_units=128, fc_units=512):
    """
    ctpn网络
    :param base_features: (B,H,W,C)
    :param num_anchors: anchors个数
    :param rnn_units:
    :param fc_units:
    :return:
    """
    # 沿着宽度方式做rnn
    rnn_forward = layers.TimeDistributed(layers.GRU(rnn_units, return_sequences=True, kernel_initializer='he_normal'),
                                         name='gru_forward')(base_features)
    rnn_backward = layers.TimeDistributed(
        layers.GRU(rnn_units, return_sequences=True, kernel_initializer='he_normal', go_backwards=True),
        name='gru_backward')(base_features)

    rnn_output = layers.Concatenate(name='gru_concat')([rnn_forward, rnn_backward])  # (B,H,W,256)

    # conv实现fc
    fc_output = layers.Conv2D(fc_units, kernel_size=(1, 1), activation='relu', name='fc_output')(
        rnn_output)  # (B,H,W,512)

    # 分类
    class_logits = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='cls')(fc_output)
    class_logits = layers.Reshape(target_shape=(-1, 2 * num_anchors), name='cls_reshape')(class_logits)
    # 中心点垂直坐标和高度回归
    predict_deltas = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='deltas')(fc_output)
    predict_deltas = layers.Reshape(target_shape=(-1, 2 * num_anchors), name='deltas_reshape')(predict_deltas)
    # 侧边精调
    predict_side_deltas = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='side_deltas')(fc_output)
    predict_side_deltas = layers.Reshape(target_shape=(-1, 2 * num_anchors), name='side_deltas_reshape')(
        predict_side_deltas)
    return class_logits, predict_deltas, predict_side_deltas
