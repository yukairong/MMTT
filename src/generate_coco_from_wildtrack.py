"""
Generates COCO data and annotation structure from WildTrack dataset
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

def generate_coco_from_wildtrack(data_root=None, split_name=None,
                                 seqs_names=None, frame_range=None):
    """
    用于将WildTrack数据转换成COCO格式的数据
    :param data_root: WildTrack数据集的保存路径
    :param split_name: 生成数据的coco保存文件夹名称
    :param seqs_names: 挑选的视角文件夹名称
    :param frame_range:
    :return:
    """
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    coco_dir = os.path.join(data_root, split_name)

    # 删除coco_dir文件夹下所有的文件
    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [
        {
            "supercategory": "person",
            "name": "person",
            "id": 1
        }
    ]
    annotations['annotations'] = []

    # 存储标注信息的文件路径
    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))

    # 如果该该标注文件夹不存在，则创建一个新的
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    # 标注信息文件
    annotation_dir = os.path.join(annotations_dir, f'{split_name}.json')

    # 图片操作
    img_id = 0

    # 将所有视角的文件夹进行便利排序
    seqs = sorted(os.listdir(data_root))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs # 将包含的视角文件信息保存在sequences中
    annotations['frame_range'] = frame_range    # 将frame_range信息保存到标注信息的frame_range中


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate COCO from WildTrack")
