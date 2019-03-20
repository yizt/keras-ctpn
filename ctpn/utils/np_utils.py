# -*- coding: utf-8 -*-
"""
   File Name：     np_utils
   Description :  numpy 工具类
   Author :       mick.yi
   date：          2019/2/19
"""

import numpy as np
from shapely.geometry import Polygon


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


def compute_iou_1vn(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
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


def non_max_suppression(boxes, scores, iou_threshold):
    """
    非极大抑制
    :param boxes: [n,(y1,x1,y2,x2)]
    :param scores: [n]
    :param iou_threshold:
    :return:
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou_1vn(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > iou_threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def quadrangle_iou(quadrangle_a, quadrangle_b):
    """
    四边形iou
    :param quadrangle_a: 一维numpy数组[(x1,y1,x2,y2,x3,y3,x4,y4)]
    :param quadrangle_b: 一维numpy数组[(x1,y1,x2,y2,x3,y3,x4,y4)]
    :return:
    """
    a = Polygon(quadrangle_a.reshape((4, 2)))
    b = Polygon(quadrangle_b.reshape((4, 2)))
    if not a.is_valid or not b.is_valid:
        return 0
    inter = Polygon(a).intersection(Polygon(b)).area
    union = a.area + b.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def quadrangle_nms(quadrangles, scores, iou_threshold):
    """
    四边形nms
    :param quadrangles: 四边形坐标，二维numpy数组[n,(x1,y1,x2,y2,x3,y3,x4,y4)]
    :param scores: 四边形得分,[n]
    :param iou_threshold: iou阈值
    :return:
    """
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        # 选择得分最高的
        i = order[0]
        keep.append(i)
        # 逐个计算iou
        overlap = np.array([quadrangle_iou(quadrangles[i], quadrangles[t]) for t in order[1:]])
        # 小于阈值的,用于下一个极值点选择
        indices = np.where(overlap < iou_threshold)[0]
        order = order[indices + 1]

    return keep


def main():
    x = np.zeros(shape=(0, 4))
    y = pad_to_fixed_size(x, 5)
    print(y.shape)
    x = np.asarray([], np.float32).reshape((-1, 4))
    y = pad_to_fixed_size(x, 5)
    print(y.shape)


if __name__ == '__main__':
    main()
