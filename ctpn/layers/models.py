# -*- coding: utf-8 -*-
"""
   File Name：     models
   Description :  模型
   Author :       mick.yi
   date：          2019/3/13
"""
import keras
from keras import layers
from keras import Input, Model
import tensorflow as tf
from .base_net import resnet50
from .anchor import CtpnAnchor
from .target import CtpnTarget
from .losses import ctpn_cls_loss, ctpn_regress_loss, side_regress_loss
from .text_proposals import TextProposal


def ctpn_net(config, stage='train'):
    # 网络构建
    # input_image = Input(batch_shape=(config.IMAGES_PER_GPU,) + config.IMAGE_SHAPE, name='input_image')
    # input_image_meta = Input(batch_shape=(config.IMAGES_PER_GPU, 12), name='input_image_meta')
    # gt_class_ids = Input(batch_shape=(config.IMAGES_PER_GPU, config.MAX_GT_INSTANCES, 2), name='gt_class_ids')
    # gt_boxes = Input(batch_shape=(config.IMAGES_PER_GPU, config.MAX_GT_INSTANCES, 5), name='gt_boxes')
    input_image = Input(shape=config.IMAGE_SHAPE, name='input_image')
    input_image_meta = Input(shape=(12, 1), name='input_image_meta')
    gt_class_ids = Input(shape=(config.MAX_GT_INSTANCES, 2), name='gt_class_ids')
    gt_boxes = Input(shape=(config.MAX_GT_INSTANCES, 5), name='gt_boxes')

    # 预测
    base_features = resnet50(input_image)
    num_anchors = len(config.ANCHORS_HEIGHT)
    predict_class_logits, predict_deltas, predict_side_deltas = ctpn(base_features, num_anchors, 64, 256)

    # anchors生成
    anchors, valid_anchors_indices = CtpnAnchor(config.ANCHORS_HEIGHT, config.ANCHORS_WIDTH, config.NET_STRIDE,
                                                config.IMAGE_SHAPE, name='gen_ctpn_anchors')(base_features)

    if stage == 'train':
        targets = CtpnTarget(config.IMAGES_PER_GPU,
                             train_anchors_num=config.TRAIN_ANCHORS_PER_IMAGE,
                             positive_ratios=config.ANCHOR_POSITIVE_RATIO,
                             max_gt_num=config.MAX_GT_INSTANCES,
                             name='ctpn_target')([gt_boxes, gt_class_ids, anchors, valid_anchors_indices])
        deltas, class_ids, anchors_indices = targets[:3]
        # 损失函数
        regress_loss = layers.Lambda(lambda x: ctpn_regress_loss(*x),
                                     name='ctpn_regress_loss')([predict_deltas, deltas, anchors_indices])
        side_loss = layers.Lambda(lambda x: side_regress_loss(*x),
                                  name='side_regress_loss')([predict_side_deltas, deltas, anchors_indices])
        cls_loss = layers.Lambda(lambda x: ctpn_cls_loss(*x),
                                 name='ctpn_class_loss')([predict_class_logits, class_ids, anchors_indices])
        model = Model(inputs=[input_image, gt_boxes, gt_class_ids],
                      outputs=[regress_loss, cls_loss, side_loss])

    else:
        text_boxes, text_scores, text_class_logits = TextProposal(config.IMAGES_PER_GPU,
                                                                  score_threshold=config.TEXT_PROPOSALS_MIN_SCORE,
                                                                  output_box_num=config.TEXT_PROPOSALS_MAX_NUM,
                                                                  iou_threshold=config.TEXT_PROPOSALS_NMS_THRESH,
                                                                  name='text_proposals')(
            [predict_deltas, predict_side_deltas, predict_class_logits, anchors, valid_anchors_indices])
        model = Model(inputs=input_image, outputs=[text_boxes, text_scores, text_class_logits])
    return model


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
    class_logits = layers.Reshape(target_shape=(-1, 2), name='cls_reshape')(class_logits)
    # 中心点垂直坐标和高度回归
    predict_deltas = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='deltas')(fc_output)
    predict_deltas = layers.Reshape(target_shape=(-1, 2), name='deltas_reshape')(predict_deltas)
    # 侧边精调(只需要预测x偏移即可)
    predict_side_deltas = layers.Conv2D(num_anchors, kernel_size=(1, 1), name='side_deltas')(fc_output)
    predict_side_deltas = layers.Reshape(target_shape=(-1, 1), name='side_deltas_reshape')(
        predict_side_deltas)
    return class_logits, predict_deltas, predict_side_deltas


def _get_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    return None


def compile(keras_model, config, loss_names=[]):
    """
    编译模型，增加损失函数，L2正则化以
    :param keras_model:
    :param config:
    :param loss_names: 损失函数列表
    :return:
    """
    # 优化目标
    optimizer = keras.optimizers.SGD(
        lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
        clipnorm=config.GRADIENT_CLIP_NORM)
    # 增加损失函数，首先清除之前的，防止重复
    keras_model._losses = []
    keras_model._per_input_losses = {}

    for name in loss_names:
        layer = _get_layer(keras_model, name)
        if layer is None or layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # 增加L2正则化
    # 跳过批标准化层的 gamma 和 beta 权重
    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        for w in keras_model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    keras_model.add_loss(tf.add_n(reg_losses))

    # 编译
    keras_model.compile(
        optimizer=optimizer,
        loss=[None] * len(keras_model.outputs))  # 使用虚拟损失

    # 为每个损失函数增加度量
    for name in loss_names:
        if name in keras_model.metrics_names:
            continue
        layer = _get_layer(keras_model, name)
        if layer is None:
            continue
        keras_model.metrics_names.append(name)
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list):
    """
    增加度量
    :param keras_model: 模型
    :param metric_name_list: 度量名称列表
    :param metric_tensor_list: 度量张量列表
    :return: 无
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        keras_model.metrics_tensors.append(tensor)
