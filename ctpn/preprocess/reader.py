# -*- coding: utf-8 -*-
"""
   File Name：     reader
   Description :
   Author :       mick.yi
   date：          2019/3/14
"""
import numpy as np
import os


def load_annotation(annotation_path, image_dir):
    """
    加载标注信息
    :param annotation_path:
    :param image_dir:
    :return:
    """
    image_annotation = {}
    # 文件名称，路径
    base_name = os.path.basename(annotation_path)
    image_name = base_name[3:-3] + 'jpg'  # gt_img_3.txt,img_3.jpg
    image_annotation["annotation_path"] = annotation_path
    image_annotation["file_name"] = image_name  # 图像文件名
    image_annotation["image_path"] = os.path.join(image_dir, image_name)
    # 读取边框标注
    bbox = []
    quadrilateral = []  # 四边形

    with open(annotation_path, "r", encoding='utf-8') as f:
        lines = f.read().encode('utf-8').decode('utf-8-sig').splitlines()
        # lines = f.readlines()
        # print(lines)
    for line in lines:
        line = line.strip().split(",")
        # 左上、右上、右下、左下 四个坐标 如：377,117,463,117,465,130,378,130
        lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = map(float, line[:8])
        x_min, y_min, x_max, y_max = min(lt_x, lb_x), min(lt_y, rt_y), max(rt_x, rb_x), max(lb_y, rb_y)
        bbox.append([y_min, x_min, y_max, x_max])
        quadrilateral.append([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y])

    image_annotation["boxes"] = np.asarray(bbox, np.float32).reshape((-1, 4))
    image_annotation["quadrilaterals"] = np.asarray(quadrilateral, np.float32).reshape((-1, 8))
    image_annotation["labels"] = np.ones(shape=(len(bbox)), dtype=np.uint8)
    return image_annotation
