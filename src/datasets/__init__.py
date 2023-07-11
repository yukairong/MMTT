from argparse import Namespace

from pycocotools.coco import COCO

from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection

from src.datasets.wildtrack import build_wildtrack


def build_dataset(split: str, args: Namespace) -> Dataset:
    """
    调用 coco.build_coco 函数，生成数据集
    :param split: 是生成"train" 还是 "val"
    :return: 数据集 Dataset
    """
    print(args)
    if args.dataset == 'wildTrack':
        dataset = build_wildtrack(split, args)
    else:
        raise ValueError(f'dataset {args.dataset} not supported')

    return dataset


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """
    Return COCO class from PyTorch dataset for evaluation with COCO eval.
    :param dataset:
    :return:
    """
    for _ in range(10):
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco
