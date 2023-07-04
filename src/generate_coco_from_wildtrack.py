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

# 存储数据集中的seqs信息
DATSET_SEQS_INFO = {
    "WildTrack" : {"img_width": 1920, "img_height": 1080, "seq_length": 401}
}

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
    # print("coco dir:", coco_dir)

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
    # print("annotation dir:", annotations_dir)

    # 如果该该标注文件夹不存在，则创建一个新的
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    # 标注信息文件
    annotation_dir = os.path.join(annotations_dir, f'{split_name}.json')
    # print("annotation dir:", annotation_dir)

    # 图片操作
    img_id = 0
    # WildTrack数据集中所有的图片保存在Image_subsets中
    imgs_data_root = os.path.join(data_root, "Image_subsets")

    # 将所有视角的文件夹进行便利排序
    seqs = sorted(os.listdir(imgs_data_root))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs  # 将包含的视角文件信息保存在sequences中
    annotations['frame_range'] = frame_range  # 将frame_range信息保存到标注信息的frame_range中
    print(split_name, seqs)

    # 对每个视角下的所有图片进行处理
    for view_id, seq in enumerate(seqs):
        # WildTrack数据集中并没有config文件,需要手动输入
        img_width = DATSET_SEQS_INFO["WildTrack"]["img_width"]
        img_height = DATSET_SEQS_INFO["WildTrack"]["img_height"]
        seq_length = DATSET_SEQS_INFO["WildTrack"]["seq_length"]

        # 便利该视角下的所有图片
        seq_list_dir = os.listdir(os.path.join(imgs_data_root, seq))

        start_frame = int(frame_range["start"] * seq_length)
        end_frame = int(frame_range["end"] * seq_length)

        # 取出起始帧到结束帧区间内的所有图片
        seq_list_dir = seq_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seq_list_dir)}/{seq_length}")
        seq_length = len(seq_list_dir)

        first_frame_image_id = -1   # 记录第一帧的图片id
        for i, img in enumerate(sorted(seq_list_dir)):

            # 标记第一帧的图像id
            if i == 0:
                first_frame_image_id = img_id

            assert first_frame_image_id >= 0, "没有找到第一帧图片id"

            annotations['images'].append({
                "file_name": f"{seq}_{img}",    # 图片的文件信息名称
                "height": img_height,   # 图片高度
                "width": img_width, # 图片宽度
                "id": img_id,   # 图片id
                "frame_id": i,  # 帧数
                "seq_length": seq_length,   # 序列的总长度(帧总数)
                "first_frame_image_id": first_frame_image_id,   # 第1帧图像的id
                "view_id": view_id  # 视角的id
            })

            img_id += 1

            # 将图片复制到创建的coco_dir文件夹中
            # 如果出现报错，请以管理员身份进行运行
            os.symlink(os.path.join(imgs_data_root, seq, img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # 对标注信息进行处理
    annotation_id = 0
    img_file_name_to_id = {
        img_dict["file_name"]: img_dict["id"]
        for img_dict in annotations["images"]
    }

    # TODO: 标注信息进行处理

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Generate COCO from WildTrack")
    data_root = r"D:\datasets\Wildtrack_dataset_full\Wildtrack_dataset"
    seqs_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    generate_coco_from_wildtrack(
        data_root=data_root, split_name="wildtrack_train_coco",
        seqs_names=seqs_names, frame_range=None
    )
