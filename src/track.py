# import os
import os
import pathlib

# import sys
# import time
# from os import path as osp
#
import motmetrics as mm  # MOT 评价指标
import numpy as np
import sacred
import torch
# import tqdm
import yaml

# from torch.utils.data import DataLoader
from src.datasets.tracking import TrackDatasetFactory
# from trackformer.models.tracker import Tracker
from src.utils.misc import nested_dict_to_namespace

# from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum, interpolate_tracks, plot_sequence)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

mm.lap.default_solver = 'lap'

ex = sacred.Experiment('track')
ex.add_config('../cfgs/track.yaml')


# ex.add_named_config('reid', '../cfgs/track_reid.yaml')


@ex.automain
def main(write_images, output_dir, _run, seed, obj_detect_checkpoint_file, _log, verbose, dataset_name,
         data_root_dir, obj_detector_model=None, ):
    """

    :param data_root_dir: 数据集的根路径
    :param dataset_name: 所用的数据集名称
    :param verbose: 是否对bbox做超出边界判断
    :param _log:
    :param obj_detect_checkpoint_file: 训练好的权重文件路径
    :param seed: 随机种子
    :param _run: 命令行中添加的命令
    :param obj_detector_model: 默认为None
    :param write_images: compile video with: `ffmpeg -f image2 -framerate 15 -i %06d.jpg -vcodec
                         libx264 -y movie.mp4 -vf scale=320:-1`
    :param output_dir: 结果保存路径
    :return:
    """
    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during training. in that case we omit verbose outputs.
    # track.yaml 没有提供这个键
    if obj_detector_model is None:
        sacred.commands.print_config(_run)

    # set all seeds, 便于复现
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        print('you should add code in there to save results.')

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    if obj_detector_model is None:
        # 在训练好的权重pth文件夹中有对应保存的config文件
        obj_detect_config_path = os.path.join(os.path.dirname(obj_detect_checkpoint_file), 'config.yaml')
        obj_detect_args = nested_dict_to_namespace(yaml.unsafe_load(open(obj_detect_config_path)))
        img_transform = obj_detect_args.img_transform

        # 需要结合自己的更改
        # obj_detector, _, obj_detector_post = build_model(obj_detect_args)
        # # 加载训练好的权重
        # obj_detect_checkpoint = torch.load(obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)
        # obj_detect_state_dict = obj_detect_checkpoint['model']
        #
        # obj_detect_state_dict = {
        #     k.replace('detr.', ''): v
        #     for k, v in obj_detect_state_dict.items() if 'track_encoding' not in k}
        #
        # obj_detector.load_state_dict(obj_detect_state_dict)
        # if 'epoch' in obj_detect_checkpoint:
        #     _log.info(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")
        #
        # obj_detector.cuda()
    else:
        obj_detector = obj_detector_model['model']
        obj_detector_post = obj_detector_model['post']
        img_transform = obj_detector_model['img_transform']

    # if hasattr(obj_detector, 'tracking'):
    #     obj_detector.tracking()

    track_logger = None

    # 是否对bbox做判断。超出边界
    if verbose:
        track_logger = _log.info

    # tracker = Tracker(obj_detector, obj_detector_post, tracker_cfg, generate_attention_maps, track_logger, verbose)

    time_total = 0
    num_frames = 0
    mot_accums = []

    # 数据集里含有7个视角下的数据
    dataset_MultiView = TrackDatasetFactory(dataset_name, root_dir=data_root_dir, img_transform=img_transform)
    _log.info(dataset_MultiView)
    _log.info(f'dataset.length = {len(dataset_MultiView[0])}.')

    for seq in dataset_MultiView:
        # tracker.reset()
        _log.info(f"------------------")
        _log.info(f"TRACK SEQ: {seq}")

    print('')
