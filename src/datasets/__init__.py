from argparse import Namespace

from torch.utils.data import Dataset

# from datasets.mot import build_mot_wildTrack
from datasets.wildtrack import build_wildtrack

def build_dataset(split: str, args: Namespace) -> Dataset:
    """
    调用 coco.build_coco 函数，生成数据集
    :param split: 是生成"train" 还是 "val"
    :return: 数据集 Dataset
    """
    print(args)
    if args.dataset == 'mot_wildTrack':
        # dataset = build_mot_wildTrack(split, args)
        dataset = build_wildtrack(split, args)
    else:
        raise ValueError(f'dataset {args.dataset} not supported')

    return dataset
