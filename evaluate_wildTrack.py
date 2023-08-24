import os
import sys
import time
from os import path as osp

import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.models import build_model
from src.tracker import Tracker, load_result, write_results
from src.utils import misc
from src.utils.misc import nested_dict_to_namespace
from src.utils.track_utils import (get_mot_accum, plot_sequence, evaluate_mot_accums)

ex = sacred.Experiment('test')
ex.add_config('./cfgs/track.yaml')
ex.add_named_config('reid', './cfgs/track_reid.yaml')


# 打印当前运行的参数和对应的值
@ex.automain
def main(write_image, output_dir, _run, seed, _config, obj_detect_checkpoint_file, _log, verbose,
         tracker_cfg, generate_attention_maps, result_path, save_txt_path, dataset_name,
         cluster_model_path, obj_detector_model=None):
    """

    :param cluster_model_path: cluster_model 的模型权重
    :param save_txt_path: 保存检测结果的路径
    :param result_path: 保存预测的文件路径
    :param generate_attention_maps:
    :param tracker_cfg: 跟踪的相关配置，包括阈值之类的
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
        print(obj_detect_args)  # 打印加载的权重对应的训练配置文件

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
        obj_detector_post = None
        print('Please provide your obj_detector_model.')

    gnn_model_checkpoint = torch.load(cluster_model_path, map_location=lambda storage, loc: storage)
    obj_detector['gnn_model'].load_state_dict(gnn_model_checkpoint['model'])
    obj_detector['gnn_model'].to('cuda:0')


    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None
    if verbose:
        track_logger = _log.info

    # ****************************** 创建跟踪的类 ***********************************************************************
    tracker = Tracker(obj_detector['track_model'], obj_detector_post, tracker_cfg, generate_attention_maps,
                      track_logger, verbose)

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

    # ****************************** 开始前向推理 ***********************************************************************
    results = load_result(result_path)

    if not results:  # 是否之前已经检测推理过
        start = time.time()

        for frame_id, (frame_data, frame_target) in enumerate(tqdm.tqdm(data_loader_val, file=sys.stdout)):
            with torch.no_grad():
                tracker.step(frame_data, frame_target)

        results = tracker.get_results()
        time_total += time.time() - start

        _log.info(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
        _log.info(f"RUNTIME: {time.time() - start :.2f} s")

        if save_txt_path is not None:
            _log.info(f"WRITE RESULTS")
            write_results(results, save_txt_path)
    else:
        _log.info("LOAD RESULTS")

    # ****************************** 开始测试 **************************************************************************
    mot_accum = get_mot_accum(results, data_loader_val)
    mot_accums.append(mot_accum)

    if verbose:  # false
        mot_events = mot_accum.mot_events
        reid_events = mot_events[mot_events['Type'] == 'SWITCH']
        match_events = mot_events[mot_events['Type'] == 'MATCH']

        switch_gaps = []
        for index, event in reid_events.iterrows():
            frame_id, _ = index
            match_events_oid = match_events[match_events['OId'] == event['OId']]
            match_events_oid_earlier = match_events_oid[
                match_events_oid.index.get_level_values('FrameId') < frame_id]

            if not match_events_oid_earlier.empty:
                match_events_oid_earlier_frame_ids = match_events_oid_earlier.index.get_level_values('FrameId')
                last_occurrence = match_events_oid_earlier_frame_ids.max()
                switch_gap = frame_id - last_occurrence
                switch_gaps.append(switch_gap)

        switch_gaps_hist = None
        if switch_gaps:
            switch_gaps_hist, _ = np.histogram(switch_gaps, bins=list(range(0, max(switch_gaps) + 10, 10)))
            switch_gaps_hist = switch_gaps_hist.tolist()

        _log.info(f'SWITCH_GAPS_HIST (bin_width=10): {switch_gaps_hist}')

    if output_dir is not None and write_image:
        _log.info("PLOT SEQ")
        plot_sequence(results, data_loader_val, osp.join(output_dir, dataset_name), write_image,
                      generate_attention_maps)

    if time_total:
        _log.info(f"RUNTIME ALL SEQS (w/o EVAL or IMG WRITE): {time_total:.2f} s for {num_frames} frames "
                  f"({num_frames / time_total:.2f} Hz)")

    if obj_detector_model is None:
        _log.info(f"EVAL:")
        summary, str_summary = evaluate_mot_accums(mot_accums, [str(s) for s in dataset_val if not s.no_gt])
        _log.info(f'\n{str_summary}')

        return summary

    return mot_accums
