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


def display_instances(image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_bbox=True,
                      colors=None, captions=None):
    """

    :param image: numpy数组，[h,w,c}
    :param boxes: 边框坐标  [num_instance,(y1,y2,x1,x2)]
    :param class_ids: [num_instances]
    :param class_names:类别名称列表或 id-名称字典
    :param scores: (optional)预测类别得分[num_instances]
    :param title: (optional)标题
    :param figsize: (optional)
    :param ax:(optional)
    :param show_bbox:(optional)
    :param colors:(optional)
    :param captions: (optional) 每个边框的说明文字
    :return:
    """
    """
    可视化实例
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

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

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
