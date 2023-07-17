from typing import Union

from torch.utils.data import ConcatDataset

DATASETS = {}

class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed
    """

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
