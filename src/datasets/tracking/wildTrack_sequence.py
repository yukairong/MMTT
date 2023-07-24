import json
import os
import os.path as osp
from argparse import Namespace

from torch.utils.data import Dataset

from src.datasets.coco import make_coco_transforms
from src.datasets.transforms import Compose


class WildTrackSequence(Dataset):
    """
    WildTrack Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    # 由于root_dir设置了数据集存储的根路径，这里只需要给出所用到的数据集所在的文件夹就行
    data_folder = 'Wildtrack_dataset'

    def __init__(self, root_dir: str = 'data', vis_threshold: float = 0.0, img_transform: Namespace = None,
                 ann_file_name: str = 'wildtrack_train_coco.json') -> None:
        """

        :param root_dir: 数据集存储的根路径 [D:\dataset\MOT]
        :param vis_threshold: Threshold of visibility of persons above which they are selected
        :param img_transform:
        """

        super().__init__()

        self._vis_threshold = vis_threshold

        self._data_dir = osp.join(root_dir, self.data_folder)  # D:\dataset\MOT\Wildtrack_dataset

        self.ann_file_path = ann_file_name

        # D:\dataset\MOT\Wildtrack_dataset\annotations\wildtrack_train_coco
        self.ann_file_path = osp.join(self._data_dir, 'annotations', self.ann_file_path)

        # 存储所有的存储标注的文件
        self.annotations = json.load(open(self.ann_file_path))

        # 这里只是简单测试，整个训练集作为测试集
        # D:\dataset\MOT\Wildtrack_dataset\Image_subsets\
        self._train_folders = os.listdir(os.path.join(self._data_dir, 'Image_subsets'))
        self._test_folders = os.listdir(os.path.join(self._data_dir, 'Image_subsets'))

        # 对验证集做val对应的变换，无需crop等操作
        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        # 获取所有视角的信息
        # self.data = self._sequence()

    @property
    def seq_length(self) -> int:
        """
            Return sequence length, i.e, number of frames.
        """
        return self.annotations['images'][0]['seq_length']

    # -----------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.seq_length

    # -----------------------------------------------------------------------------------------------------------------
    # def _sequence(self) -> List[dict]:
    #     # 所有视角的图片存储的文件夹路径
    #     img_dir = osp.join(self._data_dir, 'wildtrack_train_coco')
    #
    #     boxes, visibility = self.get_track_boxes_and_visbility()
    #
    #     total = [
    #         {'gt': boxes[i],
    #          'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
    #          'vis': visibility[i],
    #          'tra': dets[i]
    #          } for i in range(0, self.seq_length)]
    #
    #     return total
    #
    # # -----------------------------------------------------------------------------------------------------------------
    # def __getitem__(self, idx: int) -> dict:
    #     """Return the ith image converted to blob"""
    #     data = self.data[idx]
    #     img = Image.open(data['im_path']).convert("RGB")
    #     width_orig, height_orig = img.size
    #
    #     img, _ = self.transforms(img)
    #     width, height = img.size(2), img.size(1)
    #
    #     sample = {'img': img,
    #               'dets': torch.tensor([det[:4] for det in data['dets']]),
    #               'img_path': data['im_path'],
    #               'gt': data['gt'],
    #               'vis': data['vis'],
    #               'orig_size': torch.as_tensor([int(height_orig), int(width_orig)]),
    #               'size': torch.as_tensor([int(height), int(width)])}
    #
    #     return sample

    # def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
    #     """ Load ground truth boxes and their visibility."""
    #     boxes = {}
    #     visibility = {}
    #
    #     for i in range(1, self.seq_length + 1):
    #         boxes[i] = {}
    #         visibility[i] = {}
    #
    #     gt_file = self.get_gt_file_path()
    #     if not osp.exists(gt_file):
    #         return boxes, visibility
    #
    #     with open(gt_file, "r") as inf:
    #         reader = csv.reader(inf, delimiter=',')
    #         for row in reader:
    #             # class person, certainity 1
    #             if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
    #                 # Make pixel indexes 0-based, should already be 0-based (or not)
    #                 x1 = int(row[2]) - 1
    #                 y1 = int(row[3]) - 1
    #                 # This -1 accounts for the width (width of 1 x1=x2)
    #                 x2 = x1 + int(row[4]) - 1
    #                 y2 = y1 + int(row[5]) - 1
    #                 bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    #
    #                 frame_id = int(row[0])
    #                 track_id = int(row[1])
    #
    #                 boxes[frame_id][track_id] = bbox
    #                 visibility[frame_id][track_id] = float(row[8])
    #
    #     return boxes, visibility

    # @property
    # def results_file_name(self) -> str:
    #     """ Generate file name of results file. """
    #     assert self._seq_name is not None, "[!] No seq_name, probably using combined database"
    #
    #     if self._dets is None:
    #         return f"{self._seq_name}.txt"
    #
    #     return f"{self}.txt"
    #
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
    #     result_file_path = osp.join(output_dir, self.results_file_name)
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
    #
    # def load_results(self, results_dir: str) -> dict:
    #     results = {}
    #     if results_dir is None:
    #         return results
    #
    #     file_path = osp.join(results_dir, self.results_file_name)
    #
    #     if not os.path.isfile(file_path):
    #         return results
    #
    #     with open(file_path, "r") as file:
    #         csv_reader = csv.reader(file, delimiter=',')
    #
    #         for row in csv_reader:
    #             frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1
    #
    #             if track_id not in results:
    #                 results[track_id] = {}
    #
    #             x1 = float(row[2]) - 1
    #             y1 = float(row[3]) - 1
    #             x2 = float(row[4]) - 1 + x1
    #             y2 = float(row[5]) - 1 + y1
    #
    #             results[track_id][frame_id] = {}
    #             results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
    #             results[track_id][frame_id]['score'] = 1.0
    #
    #     return results
