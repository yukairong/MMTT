import os
from os import path as osp

import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.models import build_model
from src.utils import misc
from src.utils.misc import nested_dict_to_namespace

ex = sacred.Experiment('test')
ex.add_config('./cfgs/track.yaml')
# ex.add_named_config('reid', './cfgs/track_reid.yaml')


# 打印当前运行的参数和对应的值
@ex.automain
def main(write_image, output_dir, _run, seed, _config, obj_detect_checkpoint_file, _log, verbose,
         obj_detector_model=None):
    """

    :param verbose: 对超出边界的框作调整
    :param _log:
    :param obj_detect_checkpoint_file: 训练好的模型权重文件
    :param _config:
    :param seed: 随机种子，便于复现结果一致
    :param _run:
    :param obj_detector_model: 目标检测模型
    :param write_image: 是否将bbox绘制在图像上
    :param output_dir: 保存结果的文件夹路径
    :return:
    """
    if write_image:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during training. in that case we omit verbose outputs.
    if obj_detector_model is None:
        sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # 保存结果
    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        # 保存此次运行的配置文件
        yaml.dump(_config, open(osp.join(output_dir, 'track.yaml'), 'w'), default_flow_style=False)

    # *************************** Initialize the modules  ***********************************************
    if obj_detector_model is None:
        # 如果提供训练好的权重文件，则一起提供该权重对应的配置文件config.yaml
        obj_detect_config_path = os.path.join(os.path.dirname(obj_detect_checkpoint_file), 'config.yaml')
        obj_detect_args = nested_dict_to_namespace(yaml.unsafe_load(open(obj_detect_config_path)))

        # 图像变换
        img_transform = obj_detect_args.img_transform

        # 模型构建 obj_detector 包含 track_model 和 cluster_model
        obj_detector, _, obj_detector_post = build_model(obj_detect_args)

        # 加载模型权重
        obj_detect_checkpoint = torch.load(obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)
        obj_detect_state_dict = obj_detect_checkpoint['model']
        obj_detector['track_model'].load_state_dict(obj_detect_state_dict)
        obj_detector['track_model'].to('cuda:0')
        obj_detector['cluster_model'].to('cuda:0')
    else:
        obj_detect_args = None
        obj_detector = None
        print('Please provide your obj_detector_model.')

    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None
    if verbose:
        track_logger = _log.info

    # ****************************** 创建跟踪的类 ***********************************************************************
    # tracker = Tracker(obj_detector, obj_detector_post, tracker_cfg, generate_attention_maps, track_logger, verbose)

    time_total = 0
    num_frames = 0
    mot_accums = []

    # ****************************** 构建数据集 **************************************************************************
    dataset_val = build_dataset(split='val', args=obj_detect_args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=obj_detect_args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=misc.collate_fn,
        num_workers=obj_detect_args.num_workers)

    # ****************************** 开始测试 **************************************************************************
    for i, (samples, targets) in enumerate(data_loader_val):
        print(i)
        device = obj_detect_args.device
        samples = samples.to(device)
        targets = [misc.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, *_ = obj_detector['track_model'](samples, None)

    return True
