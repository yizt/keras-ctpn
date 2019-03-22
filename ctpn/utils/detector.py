# -*- coding: utf-8 -*-
"""
   File Name：     detector
   Description :   文本行检测器
   Author :       mick.yi
   date：          2019/3/14
"""
import numpy as np
from .text_proposal_connector import TextProposalConnector
from ..utils import np_utils


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self, config):
        self.config = config
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, text_proposals, scores, image_shape, window):
        """
        检测文本行
        :param text_proposals: 文本提议框
        :param scores: 文本框得分
        :param image_shape: 图像形状
        :param window [y1,x1,y2,x2] 去除padding后的窗口
        :return: text_lines; [ num,(y1,x1,y2,x2,score)]
        """

        scores = normalize(scores)  # 加上后，效果变差; 评估结果好像还是好一点
        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, image_shape)
        keep_indices = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_indices]
        text_lines = filter_out_of_window(text_lines, window)

        # 文本行nms
        if text_lines.shape[0] != 0:
            keep_indices = np_utils.quadrangle_nms(text_lines[:, :8], text_lines[:, 8],
                                                   self.config.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_indices]

        return text_lines

    def filter_boxes(self, text_lines):
        widths = text_lines[:, 2] - text_lines[:, 0]
        scores = text_lines[:, -1]
        return np.where((scores > self.config.LINE_MIN_SCORE) &
                        (widths > (self.config.TEXT_PROPOSALS_WIDTH * self.config.MIN_NUM_PROPOSALS)))[0]


def filter_out_of_window(text_lines, window):
    """
    过滤窗口外的text_lines
    :param text_lines: [n,9]
    :param window: [y1,x1,y2,x2]
    :return:
    """
    y1, x1, y2, x2 = window

    quadrilaterals = np.reshape(text_lines[:, :8], (-1, 4, 2))  # [n,4 points,(x,y)]
    min_x = np.min(quadrilaterals[:, :, 0], axis=1)  # [n]
    max_x = np.max(quadrilaterals[:, :, 0], axis=1)
    min_y = np.min(quadrilaterals[:, :, 1], axis=1)
    max_y = np.max(quadrilaterals[:, :, 1], axis=1)
    # 窗口内的text_lines
    indices = np.where(np.logical_and(np.logical_and(np.logical_and(min_x >= x1,
                                                                    max_x <= x2),
                                                     min_y >= y1),
                                      max_y <= y2))
    return text_lines[indices]
