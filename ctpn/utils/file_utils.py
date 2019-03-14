# -*- coding: utf-8 -*-
"""
   File Name：     file_utils
   Description :  文件处理工具类
   Author :       mick.yi
   date：          2019/2/19
"""
import os


def get_sub_files(dir_path, recursive=False):
    """
    获取目录下所有文件名
    :param dir_path:
    :param recursive: 是否递归
    :return:
    """
    file_paths = []
    for dir_name in os.listdir(dir_path):
        cur_dir_path = os.path.join(dir_path, dir_name)
        if os.path.isdir(cur_dir_path) and recursive:
            file_paths = file_paths + get_sub_files(cur_dir_path)
        else:
            file_paths.append(cur_dir_path)
    return file_paths
