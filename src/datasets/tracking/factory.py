from typing import Union

from torch.utils.data import ConcatDataset

from src.datasets.tracking.demo_sequence import DemoSequence
from src.datasets.tracking.wildTrack_sequence import WildTrackSequence

# 将所需可能用到的测试数据集都放在这个字典里
DATASETS = {
    'DEMO': (lambda kwargs: [DemoSequence(**kwargs), ]),
    'WildTrack': (lambda kwargs: [WildTrackSequence(**kwargs), ])
}


class TrackDatasetFactory:
    """
    A central class to manage the individual dataset loaders
    This class contains the datasets. Once initialized the individual parts (e.g. sequences) can be accessed
    """

    # datasets传进来的参数是 'WildTrack
    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """
        Initialize the corresponding dataloader
        :param datasets: the name of dataset or list of dataset names
        :param kwargs: arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
