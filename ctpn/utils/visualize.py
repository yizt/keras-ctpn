# -*- coding: utf-8 -*-
"""
   File Name：     visualize
   Description :  可视化
   Author :       mick.yi
   date：          2019/2/20
"""

import matplotlib.pyplot as plt
from matplotlib import patches
import random
import colorsys
import numpy as np


def random_colors(N, bright=True):
    """
    生成随机RGB颜色
    :param N: 颜色数量
    :param bright:
    :return:
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_boxes(image, boxes,
                  scores=None, title="",
                  figsize=(16, 16), ax=None,
                  show_bbox=True,
                  colors=None):
    """
    可视化实例
    :param image: numpy数组，[h,w,c}
    :param boxes: 边框坐标  [num_instance,(y1,y2,x1,x2)]
    :param scores: (optional)预测类别得分[num_instances]
    :param title: (optional)标题
    :param figsize: (optional)
    :param ax:(optional)
    :param show_bbox:(optional)
    :param colors:(optional)
    :return:
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('on')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
        ax.text(x1, y1 + 8, scores[i] if scores is not None else '',
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def display_polygons(image, polygons, scores=None, figsize=(16, 16), ax=None, colors=None):
    auto_show = False
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    if colors is None:
        colors = random_colors(len(polygons))

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    for i, polygon in enumerate(polygons):
        color = colors[i]
        polygon = np.reshape(polygon, (-1, 2))  # 转为[n,(x,y)]
        patch = patches.Polygon(polygon, facecolor=None, fill=False, color=color)
        ax.add_patch(patch)
        # 多边形得分
        x1, y1 = polygon[0][:]
        ax.text(x1, y1 - 1, scores[i] if scores is not None else '',
                color='w', size=11, backgroundcolor="none")
    ax.imshow(image.astype(np.uint8))
    if auto_show:
        plt.show()
