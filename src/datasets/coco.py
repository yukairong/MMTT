import random
from collections import Counter

import torch
import torch.utils.data
import torchvision

import transforms as T


# *********************************************************************************************************************

class CocoDetection(torchvision.datasets.CocoDetection):
    fields = ["labels", "area", "iscrowd", "boxes", "track_ids", "masks"]

    def __init__(self, img_folder, ann_file, transforms, norm_transforms,
                 remove_no_obj_imgs=True, prev_frame=False, prev_frame_rnd_augs=0.0, prev_prev_frame=False,
                 min_num_objects=0):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms

        annos_image_ids = [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
        if remove_no_obj_imgs:
            self.ids = sorted(list(set(annos_image_ids)))

        if min_num_objects:
            counter = Counter(annos_image_ids)

            self.ids = [i for i in self.ids if counter[i] >= min_num_objects]

        self._prev_frame = prev_frame
        self._prev_frame_rnd_augs = prev_frame_rnd_augs
        self._prev_prev_frame = prev_prev_frame

    def _getitem_from_id(self, image_id, random_state=None, random_jitter=True):
        # if random state is given we do the data augmentation with the state
        # and then apply the random jitter. this ensures that (simulated) adjacent
        # frames have independent jitter.
        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        img, target = super(CocoDetection, self).__getitem__(image_id)
        image_id = self.ids[image_id]
        target = {'image_id': image_id,
                  'annotations': target}
        img, target = self.prepare(img, target)

        if 'track_ids' not in target:
            target['track_ids'] = torch.arange(len(target['labels']))

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ignore
        ignore = target.pop("ignore").bool()
        for field in self.fields:
            if field in target:
                target[f"{field}_ignore"] = target[field][ignore]
                target[field] = target[field][~ignore]

        if random_state is not None:
            random.setstate(curr_random_state['random'])
            torch.random.set_rng_state(curr_random_state['torch'])

        if random_jitter:
            img, target = self._add_random_jitter(img, target)
        img, target = self._norm_transforms(img, target)

        return img, target

    # TODO: add to the transforms and merge norm_transforms into transforms
    def _add_random_jitter(self, img, target):
        if self._prev_frame_rnd_augs:
            orig_w, orig_h = img.size

            crop_width = random.randint(
                int((1.0 - self._prev_frame_rnd_augs) * orig_w),
                orig_w)
            crop_height = int(orig_h * crop_width / orig_w)

            transform = T.RandomCrop((crop_height, crop_width))
            img, target = transform(img, target)

            img, target = T.resize(img, target, (orig_w, orig_h))

        return img, target

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}
        img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

        if self._prev_frame:
            # PREV
            prev_img, prev_target = self._getitem_from_id(idx, random_state)
            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV
                prev_prev_img, prev_prev_target = self._getitem_from_id(idx, random_state)
                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, *args):
        pass


# *********************************************************************************************************************
def make_coco_transforms(image_set, img_transform=None, overflow_boxes=False):
    """
    对数据集做变换
    :param image_set: 'train' or 'val'
    :param img_transform:
    :param overflow_boxes: 对 box 做限制，防止超出图像边界
    :return:
    """

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # default
    max_size = 1333
    val_width = 800
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    random_resizes = [400, 500, 600]
    random_size_crop = (384, 600)

    if img_transform is not None:
        # img_transform 只有两个参数，max_size 和 val_width
        scale = img_transform.max_size / max_size
        max_size = img_transform.max_size
        val_width = img_transform.val_width

        # scale all with respect to custom max_size
        scales = [int(scale * s) for s in scales]
        random_resizes = [int(scale * s) for s in random_resizes]
        random_size_crop = [int(scale * s) for s in random_size_crop]

    if image_set == 'train':
        transforms = [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(random_resizes),
                    T.RandomSizeCrop(*random_size_crop, overflow_boxes=overflow_boxes),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
        ]
    elif image_set == 'val':
        transforms = [T.RandomResize([val_width], max_size=max_size)]
    else:
        ValueError(f'unknown {image_set}')

    return T.Compose(transforms), normalize
