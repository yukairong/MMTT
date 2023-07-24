""""
    Wildtrack dataset
"""
import os
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.coco import make_coco_transforms
from src.datasets.transforms import Compose


class DemoSequence(Dataset):
    def __init__(self, root_dir: str = 'data', img_transform: Namespace = None) -> None:
        """

        :param root_dir: wildtrack数据集的根路径
        :param img_transform: 对图像的转换操作
        """

        super().__init__()

        self._data_dir = Path(root_dir)
        assert self._data_dir.is_dir(), f'data_root_dir:{root_dir} does not exist.'

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))
        self.data = self._sequence()  # 数据集的所有帧图片路径，是一个list，通过im_path这个key获取
        self.no_gt = True

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self._data_dir.name

    def __getitem__(self, idx: int) -> dict:
        data = self.data[idx]
        # 根据数据集中的 im_path 读取对应的图片路径
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        sample = {'img': img,
                  'img_path': data['im_path'],
                  'dets': torch.tensor([]),
                  'orig_size': torch.as_tensor([int(height_orig), int(width_orig)]),
                  'size': torch.as_tensor([int(height), int(width)])
                  }

        return sample

    def _sequence(self) -> List[dict]:
        """
        获取当前系列下所有帧图片的路径，存储在一个list中，通过key im_path 获取
        :return:
        """
        total = []
        for filename in sorted(os.listdir(self._data_dir)):
            extension = os.path.splitext(filename)[1]
            if extension in ['.png', '.jpg']:
                total.append({'im_path': os.path.join(self._data_dir, filename)})

        return total

    # def load_results(self, results_dir: str) -> dict:
    #     return {}

    # def write_results(self, results: dict, output_dir: str) -> None:
    #     """Write the tracks in the format for MOT16/MOT17 sumbission
    #
    #     results: dictionary with 1 dictionary for every track with
    #              {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num
    #
    #     Each file contains these lines:
    #     <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    #     """
    #
    #     # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     result_file_path = os.path.join(output_dir, self._data_dir.name)
    #
    #     with open(result_file_path, "w") as r_file:
    #         writer = csv.writer(r_file, delimiter=',')
    #
    #         for i, track in results.items():
    #             for frame, data in track.items():
    #                 x1 = data['bbox'][0]
    #                 y1 = data['bbox'][1]
    #                 x2 = data['bbox'][2]
    #                 y2 = data['bbox'][3]
    #
    #                 writer.writerow([
    #                     frame + 1,
    #                     i + 1,
    #                     x1 + 1,
    #                     y1 + 1,
    #                     x2 - x1 + 1,
    #                     y2 - y1 + 1,
    #                     -1, -1, -1, -1])
