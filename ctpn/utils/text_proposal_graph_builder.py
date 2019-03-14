# -*- coding: utf-8 -*-
"""
   File Name：     text_proposal_graph_builder
   Description :   文本提议框 图构建；构建文本框对
   Author :       mick.yi
   date：          2019/3/12
"""
import numpy as np


class Graph(object):
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        """
        根据图对生成文本行
        :return: list of list; 文本行列表，每个文本行是文本框索引号列表
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder(object):
    """
        文本框的配对构建
    """

    def __init__(self, max_horizontal_gap=50, min_vertical_overlaps=0.7, min_size_similarity=0.7):
        """

        :param max_horizontal_gap: 文本行内，文本框最大水平距离,超出此距离的文本框属于不同的文本行
        :param min_vertical_overlaps：文本框最小垂直Iou
        :param min_size_similarity: 文本框尺寸最小相似度
        """

        self.max_horizontal_gap = max_horizontal_gap
        self.min_vertical_overlaps = min_vertical_overlaps
        self.min_size_similarity = min_size_similarity
        self.text_proposals = None
        self.scores = None
        self.im_size = None
        self.heights = None
        self.boxes_table = None

    def get_successions(self, index):
        """
        获取指定索引号文本框的后继文本框
        :param index: 文本框索引号
        :return: 所有后继文本框的索引号列表
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[1]) + 1, min(int(box[1]) + self.max_horizontal_gap + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        """
        获取指定索引号文本框的前驱文本框
        :param index: 文本框索引号
        :return: 所有前驱文本框的索引号列表
        """
        box = self.text_proposals[index]
        results = []
        # 向前遍历
        for left in range(int(box[1]) - 1, max(int(box[1] - self.max_horizontal_gap), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        """
        是否是配对的文本框
        :param index: 文本框索引号
        :param succession_index: 后继文本框索引号,注：此文本框是后继文本框中
        :return:
        """
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        """
        两个文本框是否满足垂直方向iou条件
        :param index1:
        :param index2:
        :return: True or False
        """

        def overlaps_v(idx1, idx2):
            """
            两个边框垂直方向的iou
            """
            # 边框高宽
            h1 = self.heights[idx1]
            h2 = self.heights[idx2]
            # 垂直方向的交集
            max_y1 = max(self.text_proposals[idx2][0], self.text_proposals[idx1][0])
            min_y2 = min(self.text_proposals[idx2][2], self.text_proposals[idx1][2])
            return max(0, min_y2 - max_y1) / min(h1, h2)

        def size_similarity(idx1, idx2):
            """
            两个边框高度尺寸相似度
            """
            h1 = self.heights[idx1]
            h2 = self.heights[idx2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= self.min_vertical_overlaps and \
               size_similarity(index1, index2) >= self.min_size_similarity

    def build_graph(self, text_proposals, scores, im_size):
        """
        根据文本框构建文本框对
        :param text_proposals: 文本框，numpy 数组，[n,(y1,x1,y2,x2)]
        :param scores: 文本框得分，[n]
        :param im_size: 图像尺寸,tuple(H,W,C)
        :return: 返回二维bool类型 numpy数组，[n,n]；指示文本框两两之间是否配对
        """
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 2] - text_proposals[:, 0]  # 所有文本框的高宽

        # 安装每个文本框左侧坐标x1分组
        im_width = self.im_size[1]
        boxes_table = [[] for _ in range(im_width)]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[1])].append(index)
        self.boxes_table = boxes_table

        # 构建文本对,numpy数组[N,N]，bool类型;如果
        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            # 获取当前文本框(Bi)的后继文本框
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 后继文本框中得分最高的那个，记做Bj
            succession_index = successions[np.argmax(scores[successions])]
            # 获取Bj的前驱文本框
            precursors = self.get_precursors(succession_index)
            # print("{},{},{}".format(index, succession_index, precursors))
            # 如果Bi也是,也是Bj的前驱文本框中，得分最高的那个；则Bi,Bj构成文本框对
            if self.scores[index] >= np.max(self.scores[precursors]):
                graph[index, succession_index] = True
        return Graph(graph)
