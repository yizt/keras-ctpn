# -*- coding: utf-8 -*-
"""
   File Name：     image
   Description :  图像处理工具类
   Author :       mick.yi
   date：          2019/2/18
"""
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import random


def load_image(image_path):
    """
    加载图像
    :param image_path: 图像路径
    :return: [h,w,3] numpy数组
    """
    image = plt.imread(image_path)
    # 灰度图转为RGB
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1, 1, 3))
    elif image.shape[-1] == 1:
        image = skimage.color.gray2rgb(image)  # io.imread 报ValueError: Input image expected to be RGB, RGBA or gray
    # 标准化为0~255之间
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    # 删除alpha通道
    return image[..., :3]


def load_image_gt(image_id, image_path, output_size, gt_boxes=None,
                  gt_quadrilaterals=None, horizontal_flip=False, random_crop=False):
    """
    加载图像，生成训练输入大小的图像，并调整GT 边框，返回相关元数据信息
    :param image_id: 图像编号id
    :param image_path: 图像路径
    :param output_size: 图像输出尺寸，及网络输入到高度或宽度(默认长宽相等)
    :param gt_boxes: GT 边框 [N,(y1,x1,y2,x2)]
    :param gt_quadrilaterals:
    :param horizontal_flip: 是否水平翻转
    :param random_crop: 是否随机裁剪
    :return:
    image: (H,W,3)
    image_meta: 元数据信息，详见compose_image_meta
    gt_boxes：图像缩放及padding后对于的GT 边框坐标 [N,(y1,x1,y2,x2)]
    """
    # 加载图像
    image = load_image(image_path)
    # 随机裁剪
    if random_crop and random.random() >= 0.5:
        min_x, max_x = np.min(gt_quadrilaterals[:, ::2]), np.max(gt_quadrilaterals[:, ::2])
        min_y, max_y = np.min(gt_quadrilaterals[:, 1::2]), np.max(gt_quadrilaterals[:, 1::2])
        image, crop_window = crop_image(image, [min_y, min_x, max_y, max_x])
        # print(image.shape,[min_y, min_x, max_y, max_x],crop_window)
        # gt坐标偏移
        if gt_quadrilaterals is not None and gt_quadrilaterals.shape[0] > 0:
            gt_quadrilaterals[:, 1::2] -= crop_window[0]
            gt_quadrilaterals[:, ::2] -= crop_window[1]
    # 水平翻转
    if horizontal_flip:
        image = image[:, ::-1, :]
        # gt翻转
        if gt_quadrilaterals is not None and gt_quadrilaterals.shape[0] > 0:
            gt_quadrilaterals[:, ::2] = image.shape[1] - gt_quadrilaterals[:, ::2]
            lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = np.split(gt_quadrilaterals, 8, axis=1)
            gt_quadrilaterals = np.concatenate([rt_x, rt_y, lt_x, lt_y, lb_x, lb_y, rb_x, rb_y], axis=1)

    original_shape = image.shape
    # resize图像，并获取相关元数据信息
    image, window, scale, padding = resize_image(image, output_size)

    # 组合元数据信息
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale)
    # 根据缩放及padding调整GT边框
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        gt_boxes = adjust_box(gt_boxes, padding, scale)
    if gt_quadrilaterals is not None and gt_quadrilaterals.shape[0] > 0:
        gt_quadrilaterals = adjust_quadrilaterals(gt_quadrilaterals, padding, scale)

    return image, image_meta, gt_boxes, gt_quadrilaterals


def crop_image(image, gt_window):
    h, w = list(image.shape)[:2]
    y1, x1, y2, x2 = gt_window
    gaps = np.array([y1, x1, h - y2, w - x2])
    wy1 = np.random.randint(min(y1+1, h // 5))
    wx1 = np.random.randint(min(x1+1, w // 5))
    wy2 = h - np.random.randint(min(h - y2 + 1, h // 5))
    wx2 = w - np.random.randint(min(w - x2 + 1, w // 5))
    return image[wy1:wy2, wx1:wx2], [wy1, wx1, wy2, wx2]


def resize_image(image, max_dim):
    """
    缩放图像为正方形，指定长边大小，短边padding;
    :param image: numpy 数组(H,W,3)
    :param max_dim: 长边大小
    :return: 缩放后的图像,元素图像的宽口位置，缩放尺寸，padding
    """
    image_dtype = image.dtype
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)  # 缩放尺寸
    image = transform.resize(image, (round(h * scale), round(w * scale)),
                             order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    h, w = image.shape[:2]
    # 计算padding
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    # 原始图像在缩放图像上的窗口位置
    window = (top_pad, left_pad, h + top_pad, w + left_pad)  #
    return image.astype(image_dtype), window, scale, padding


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """
    组合图像元数据信息，返回numpy数据
    :param image_id:
    :param original_image_shape: 原始图像形状，tuple(H,W,3)
    :param image_shape: 缩放后图像形状tuple(H,W,3)
    :param window: 原始图像在缩放图像上的窗口位置（y1,x1,y2,x2)
    :param scale: 缩放因子
    :return:
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale]  # size=1
    )
    return meta


def parse_image_meta(meta):
    """
    解析图像元数据信息,注意输入是元数据信息数组
    :param meta: [12]
    :return:
    """
    image_id = meta[0]
    original_image_shape = meta[1:4]
    image_shape = meta[4:7]
    window = meta[7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }


def batch_parse_image_meta(meta):
    """
    解析图像元数据信息,注意输入是元数据信息数组
    :param meta: [batch,12]
    :return:
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }


def adjust_box(boxes, padding, scale):
    """
    根据填充和缩放因子，调整boxes的值
    :param boxes: numpy 数组; GT boxes [N,(y1,x1,y2,x2)]
    :param padding: [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    :param scale: 缩放因子
    :return:
    """
    boxes = boxes * scale
    boxes[:, 0::2] += padding[0][0]  # 高度padding
    boxes[:, 1::2] += padding[1][0]  # 宽度padding
    return boxes


def adjust_quadrilaterals(quadrilaterals, padding, scale):
    """
    根据填充和缩放因子，调整四边形的值
    :param quadrilaterals: numpy 数组; GT quadrilaterals[N,(x1,y1,x2,y2,x3,y3,x4,y4)]
    :param padding: [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    :param scale: 缩放因子
    :return:
    """
    quadrilaterals = quadrilaterals * scale
    quadrilaterals[:, 1::2] += padding[0][0]  # 高度padding
    quadrilaterals[:, 0::2] += padding[1][0]  # 宽度padding
    return quadrilaterals


def recover_detect_boxes(boxes, window, scale):
    """
    将检测边框映射到原始图像上，去除padding和缩放
    :param boxes: numpy数组，[n,(y1,x1,y2,x2)]
    :param window: [(y1,x1,y2,x2)]
    :param scale: 标量
    :return:
    """
    # 去除padding
    boxes[:, 0::2] -= window[0]
    boxes[:, 1::2] -= window[1]
    # 还原缩放
    boxes /= scale
    return boxes


def recover_detect_quad(boxes, window, scale):
    """
    将检测四边形映射到原始图像上，去除padding和缩放
    :param boxes: numpy数组，[n,(x1,y1,x2,y2,x3,y3,x4,y4)]
    :param window: [(y1,x1,y2,x2)]
    :param scale: 标量
    :return:
    """
    # 去除padding
    boxes[:, 1::2] -= window[0]  # 高度
    boxes[:, 0::2] -= window[1]  # 宽度
    # 还原缩放
    boxes /= scale
    return boxes
