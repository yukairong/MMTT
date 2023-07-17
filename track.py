import os
import sys
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from src.models import build_model
from src.utils.misc import nested_dict_to_namespace

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

mm.lap.default_solver = 'lap'

ex = sacred.Experiment("track")
ex.add_config("../cfgs/track.yaml")

@ex.automain
def main(seed, dataset_name, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate, verbose, load_results_dir,
         data_root_dir, generate_attention_maps, frame_range,
         _config, _log, _run, obj_detection_model=None):

    # 写入图片
    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during training. In that case we omit verbose outputs
    if obj_detection_model is None:
        sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        yaml.dump(
            _config,
            open(osp.join(output_dir, "track.yaml"), "w"),
            default_flow_style=False
        )
