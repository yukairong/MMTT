from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

class WildTrackDataset(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(WildTrackDataset, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range

    @property
    def sequence(self):
        return self.coco.dataset["sequences"]

def build_wildtrack(image_set, args):
    """
    构建wildtrack数据集
    :param image_set: 数据集类型, [train, val]
    :param args: config参数
    :return:
    """
    root = Path(args.wildtrack_path_train)
    assert root.exists(), f'provided COCO path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f'annotations/{split}.json'

    prev_frame_range = 0
    prev_frame_rnd_augs = 0
    if image_set == "train":
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range = args.track_prev_frame_range
    elif image_set == "val":
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1

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
    # print("data:", dataset.coco.imgs)
    print("sequence:", dataset.sequence)

