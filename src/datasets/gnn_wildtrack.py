import csv
import os
import random
from pathlib import Path
import random

import dgl
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from models.multi_view_deformable_tracking import MultiViewDeformableTrack

import utils.misc
from datasets.coco import CocoDetection, make_coco_transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_gnn_wildtrack(image_set, feature_extractor, args):
    """
    构建wildtrack的gnn训练数据集
    :param image_set: 数据集类型, [train, val]
    :param args:
    :return:
    """
    root = Path(args.wildtrack_path_train)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # TODO: 缺少验证数据，暂时全部按照训练集
    image_set = 'train'
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
        # 前一帧图像的范围,不仅仅是t-1帧,可能是t-5帧
        prev_frame_range = args.track_prev_frame_range
    elif image_set == "eval":
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1

    # 对数据集做变换 img_transform 只有两个参数, max_size 和 val_width
    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes
    )

    wild_track_dataset = WildTrackDatset(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame
    )

    dataset = WildTrackGnnDataset(
        coco_dataset=wild_track_dataset,
        feature_extractor=feature_extractor,
        device=args.device,
        mode=image_set
    )

    return dataset

class WildTrackDatset(CocoDetection):
    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(WildTrackDatset, self).__init__(*args, **kwargs)

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
            'torch': torch.random.get_rng_state()
        }

        img, target = self._getitem_from_id(
            idx, random_state, random_jitter=False
        )

        view_id = target['view_id'][0] + 1
        views_frame_image_ids = target['views_frame_image_ids']

        frame_offset = int(target['image_id'].cpu().numpy()) % 400

        res_img_list = []
        res_target_list = []
        res_view_list = []

        views_img_ids = [idx]
        for view_name, view_first_frame_img_id in views_frame_image_ids.items():
            if f'C{str(int(view_id))}' == view_name:
                continue
            views_img_ids.append(view_first_frame_img_id + frame_offset)

        for view_img_id in views_img_ids:
            view_img_id = int(view_img_id)
            img, target = self._getitem_from_id(
                view_img_id, random_state, random_jitter=False
            )

            if self._prev_frame:
                frame_id = self.coco.imgs[view_img_id]['frame_id']

                # 如果当前帧有prev_image，就从一定范围内获取前一帧 eg，t-5，并防止越界
                prev_frame_id = random.randint(
                    max(0, frame_id - self._prev_frame_range),
                    min(frame_id + self._prev_frame_range, self.seq_length(view_img_id) - 1))
                prev_image_id = self.coco.imgs[view_img_id]['first_frame_image_id'] + prev_frame_id

                prev_img, prev_target = self._getitem_from_id(
                    prev_image_id, random_state)
                target[f'prev_image'] = prev_img
                target[f'prev_target'] = prev_target

                if self._prev_prev_frame:
                    # PREV PREV frame equidistant as prev_frame
                    prev_prev_frame_id = min(max(0, prev_frame_id + prev_frame_id - frame_id),
                                             self.seq_length(view_img_id) - 1)
                    prev_prev_image_id = self.coco.imgs[view_img_id]['first_frame_image_id'] + \
                        prev_prev_frame_id

                    prev_prev_img, prev_prev_target = self._getitem_from_id(
                        prev_prev_image_id, random_state)
                    target[f'prev_prev_image'] = prev_prev_img
                    target[f'prev_prev_target'] = prev_prev_target

            res_img_list.append(img)
            res_target_list.append(target)
            res_view_list.append(view_img_id // 400)

        return res_img_list, res_target_list, res_view_list

class BaseGraphDataset(Dataset):
    """Base class for Graph Dataset"""

    def __init__(self, mode: str, feature_extractor):
        assert mode in ("train", "eval", "test")

        self.mode = mode
        self.device = 'cuda'
        self.feature_extractor = feature_extractor

        # ==== These values can be loaded via self.load_dataset() ====
        self._H = []    # homography matrices, H[seq_id][cam_id] => torch.Tensor(3*3)
        self._P = []    # images name pattern, F[seq_id][cam_id] => image path pattern
        self._S = []    # frames in sequences, S[seq_id] => frame based dict (key type: str)
        self._SFI = None    # a (N*2) size tensor, store < seq_id, frame_id>

    def __len__(self):
        raise NotImplementedError


def get_value_by_node_id(frame_dict, node_id):
    for key, val in frame_dict.items():
        view_id = key
        for frame_node_id in val['node_id']:
            if frame_node_id == node_id:
                return view_id, val['labels'], val['features']


class WildTrackGnnDataset(BaseGraphDataset):
    def __init__(self, coco_dataset, feature_extractor, device, mode):
        super(WildTrackGnnDataset, self).__init__(feature_extractor=feature_extractor, mode=mode)

        self.coco_dataset = coco_dataset
        self.feature_extractor = feature_extractor
        self.device = device

    def __getitem__(self, index):
        u, v, lbls = [], [], []

        samples, targets, views = self.coco_dataset[index]
        # samples = torch.vstack(samples).to(self.device)
        samples = [sample.to(self.device) for sample in samples]
        targets = [utils.misc.nested_dict_to_device(t, device=self.device) for t in targets]

        # track model的正向推理
        out, _, features, memory, hs = self.feature_extractor.inference(samples)
        outputs_without_aux = {
            k: v for k, v in out.items() if 'aux_outputs' not in k
        }
        indices = self.feature_extractor._matcher(outputs_without_aux, targets)

        frame_obj = {}
        node_num = 0
        cam_list = []
        node_feature = []

        for j, (target, indice) in enumerate(zip(targets, indices)):
            out_ind, target_ind = indice
            cam_id = views[j]
            cam_list += [cam_id] * len(out_ind)

            for out_i, target_i in zip(out_ind, target_ind):
                obj_feat = hs[-1, j, out_i, :]
                obj_label = target['track_ids'][target_i]

                if views[j] not in frame_obj:
                    frame_obj[views[j]] = {
                        'features': [],
                        'labels': [],
                        'node_id': []
                    }
                frame_obj[views[j]]['features'].append(obj_feat)
                frame_obj[views[j]]['labels'].append(obj_label)
                frame_obj[views[j]]['node_id'].append(node_num)

                node_feature.append(obj_feat)

                node_num += 1

        for n1 in range(node_num):
            src_cid, src_tid, _ = get_value_by_node_id(frame_obj, n1)
            for n2 in range(n1 + 1, node_num):
                dst_cid, dst_tid, _ = get_value_by_node_id(frame_obj, n2)
                if dst_cid != src_cid:
                    u.append(n1)
                    v.append(n2)
                    lbls.append(1 if dst_tid == src_tid else 0)

        graph = dgl.graph((u + v, v + u), idtype=torch.int32, device=self.device)
        graph.ndata['cam'] = torch.tensor(cam_list, dtype=torch.int32).to(self.device)
        # node_feature = torch.tensor(node_feature, dtype=torch.float32).to(self.device)
        node_feature = torch.tensor(
            [f.cpu().detach().numpy() for f in node_feature], dtype=torch.float32
        ).to(self.device)

        y_true = torch.tensor(lbls + lbls, dtype=torch.float32, device=self.device)
        embedding = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]),
            torch.cosine_similarity(node_feature[u], node_feature[v])
        )).T
        edge_feature = torch.cat((embedding, embedding))    # (E, 2)

        return graph, node_feature, edge_feature, y_true

    def __len__(self):
        return len(self.coco_dataset)

