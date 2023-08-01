import csv
import os
import random
from pathlib import Path

import torch

from src.datasets.coco import CocoDetection, make_coco_transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# *********************************************************************************************************************
def build_wildtrack(image_set, args):
    """
    构建 wildtrack 数据集
    :param image_set: 数据集类型, [train, val]
    :param args: config参数
    :return:
    """
    root = Path(args.wildtrack_path_train)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # 获取wildTrack数据集转换成COCO格式后的train val 的实际文件夹名称
    # train_split: Wildtrack_train_coco
    # val_split: Wildtrack_train_cross_val_frame_0_5_to_1_0_coco
    split = getattr(args, f"{image_set}_split")

    # wildTrack图片路径
    img_folder = root / split
    # wildTrack标注路径
    ann_file = root / f'annotations/{split}.json'

    prev_frame_range = 0
    prev_frame_rnd_augs = 0
    if image_set == "train":
        # 对前一帧图像作随机增强
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        # 前一帧图像的范围，不仅仅是 t-1 帧，可能是 t-5 帧
        prev_frame_range = args.track_prev_frame_range
    elif image_set == "val":
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1

    # 对数据集做变换 img_transform 只有两个参数，max_size 和 val_width
    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes
    )

    dataset = WildTrackDataset(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame
    )
    # print("data:", dataset.__getitem__(0))
    # print("sequence:", dataset.sequence)

    return dataset


# *********************************************************************************************************************
class WildTrackDataset(CocoDetection):
    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(WildTrackDataset, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range

    @property
    def sequence(self):
        return self.coco.dataset["sequences"]

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}

        img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

        # ############# 修改部分 ######################
        view_id = img['view_id'] + 1   # 获取当前图像的视角
        frame_offset = img['frame_id']  # 获取当前图像所在视角的帧数量
        views_frame_image_ids = img['views_frame_image_ids']    # 获取每个视角第一帧图像id

        views_img_ids = [idx]
        for view_name, view_first_frame_img_id in views_frame_image_ids.items():
            if f"C{str(view_id)}" == view_name:
                continue
            views_img_ids.append(view_first_frame_img_id + frame_offset)

        res_img_list = []
        res_target_list = []

        for view_img_id in views_img_ids:
            img, target = self._getitem_from_id(view_img_id, random_state, random_jitter=False)

            if self._prev_frame:
                frame_id = self.coco.imgs[view_img_id]['frame_id']

                # 如果当前帧有prev_image，就从一定范围内获取前一帧 eg，t-5，并防止越界
                prev_frame_id = random.randint(
                    max(0, frame_id - self._prev_frame_range),
                    min(frame_id + self._prev_frame_range, self.seq_length(view_img_id) - 1))
                prev_image_id = self.coco.imgs[view_img_id]['first_frame_image_id'] + prev_frame_id

                prev_img, prev_target = self._getitem_from_id(prev_image_id, random_state)
                target[f'prev_image'] = prev_img
                target[f'prev_target'] = prev_target

                if self._prev_prev_frame:
                    # PREV PREV frame equidistant as prev_frame
                    prev_prev_frame_id = min(max(0, prev_frame_id + prev_frame_id - frame_id),
                                             self.seq_length(view_img_id) - 1)
                    prev_prev_image_id = self.coco.imgs[view_img_id]['first_frame_image_id'] + prev_prev_frame_id

                    prev_prev_img, prev_prev_target = self._getitem_from_id(prev_prev_image_id, random_state)
                    target[f'prev_prev_image'] = prev_prev_img
                    target[f'prev_prev_target'] = prev_prev_target

            res_img_list.append(img)
            res_target_list.append(target)

        return res_img_list, res_target_list
        # ############# 修改部分 ######################

        # if self._prev_frame:
        #     frame_id = self.coco.imgs[idx]['frame_id']
        #
        #     # 如果当前帧有prev_image，就从一定范围内获取前一帧 eg，t-5，并防止越界
        #     prev_frame_id = random.randint(
        #         max(0, frame_id - self._prev_frame_range),
        #         min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1))
        #     prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + prev_frame_id
        #
        #     prev_img, prev_target = self._getitem_from_id(prev_image_id, random_state)
        #     target[f'prev_image'] = prev_img
        #     target[f'prev_target'] = prev_target
        #
        #     if self._prev_prev_frame:
        #         # PREV PREV frame equidistant as prev_frame
        #         prev_prev_frame_id = min(max(0, prev_frame_id + prev_frame_id - frame_id), self.seq_length(idx) - 1)
        #         prev_prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + prev_prev_frame_id
        #
        #         prev_prev_img, prev_prev_target = self._getitem_from_id(prev_prev_image_id, random_state)
        #         target[f'prev_prev_image'] = prev_prev_img
        #         target[f'prev_prev_target'] = prev_prev_target
        #
        # return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)
