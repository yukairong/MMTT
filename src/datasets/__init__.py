from torch.utils.data import Dataset

from coco import build_coco


def build_dataset(split: str) -> Dataset:
    """
    调用 coco.build_coco 函数，生成数据集
    :param split: 是生成"train" 还是 "val"
    :return: 数据集 Dataset
    """

    dataset = build_coco(split)

    return dataset
