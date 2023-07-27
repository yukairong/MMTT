import logging
import math
import os
import sys
from typing import Iterable

import dgl
import numpy as np
import torch

from src.utils import misc as utils
from .datasets import get_coco_api_from_dataset


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args):
    vis_iter_metrics = None

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter=" ",
        vis=vis_iter_metrics,
        debug=args.debug
    )
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
        # print(i)
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        # in order to be able to modify targets inside the forward call we need
        # to pass it through as torch.nn.parallel.DistributedDataParallel only
        # passes copies 原始图片 标注
        outputs, targets, *_ = model(samples, targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict  # 损失权重
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], lr_backbone=optimizer.param_groups[1]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_cluster_model_one_epoch(backbone: torch.nn.Module,
                                  model: torch.nn.Module,
                                  data_loader: Iterable,
                                  instance_criterion: torch.nn.Module,
                                  cluster_criterion: torch.nn.Module,
                                  optimizer: torch.optim.Optimizer,
                                  device: torch.device,
                                  epoch: int,
                                  queries_num: int):
    backbone.eval()

    model.train()
    instance_criterion.train()
    cluster_criterion.train()

    batch_size = instance_criterion.batch_size // queries_num
    loss_epoch = 0
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        # track model的正向推理过程
        x_i, x_j = backbone.decoder_forward(samples, targets)

        optimizer.zero_grad()

        x_i = x_i.to(device)
        x_j = x_j.to(device)

        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = instance_criterion(z_i, z_j)
        loss_cluster = cluster_criterion(c_i, c_j)
        loss = loss_instance + loss_cluster

        loss.backward()
        optimizer.step()

        print(
            f"Epoch[{epoch}] Step[{i}/{len(data_loader)}]  -loss_instance: {loss_instance.item()}"
            f"  -loss_cluster: {loss_cluster.item()}"
        )
        loss_epoch += loss.item()

    return loss_epoch


def train_gnn_model_one_epoch(track_model: torch.nn.Module,
                              gnn_model: torch.nn.Module,
                              data_loader: Iterable,
                              criterion: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              device: torch.device,
                              epoch: int):
    track_model.eval()
    gnn_model.train()
    criterion.train()
    loss_epoch = 0
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]
        out, _, features, memory, hs = track_model.inference(samples)
        outputs_without_aux = {
            k: v for k, v in out.items() if 'aux_outputs' not in k
        }
        indices = track_model._matcher(outputs_without_aux, targets)

        frame_obj = {}  # 存储t帧下每个视角的对象特征以及真实track id
        node_num = 0  # 节点总数
        for j, (target, indice) in enumerate(zip(targets, indices)):
            out_ind, target_ind = indice
            for out_i, target_i in zip(out_ind, target_ind):
                obj_feat = hs[-1, j, out_i, :]
                obj_label = target['track_ids'][target_i]

                if j not in frame_obj:
                    frame_obj[j] = {
                        'features': [],
                        'labels': [],
                        'node_id': []
                    }
                frame_obj[j]['features'].append(obj_feat)
                frame_obj[j]['labels'].append(obj_label)
                frame_obj[j]['node_id'].append(node_num)
                node_num += 1

        src = []  # 节点起始位
        dst = []  # 节点终点位
        node_objs = np.arange(node_num)  # 所有的节点id
        # 构建全连接的节点图
        for src_node_id in node_objs:
            for dst_node_id in node_objs:
                if src_node_id == dst_node_id:
                    continue
                src.append(src_node_id)
                dst.append(dst_node_id)

        graph = dgl.graph((src, dst)).to(device)

        node_features = torch.zeros(size=(node_num, gnn_model.in_feats)).to(device)  # 存储每个节点的特征
        edge_label = torch.zeros(size=(np.arange(node_num).sum() * 2,)).to(device)
        # 将所有视角上的目标都放置在一张图中
        for edge_indx, (src_node_id, dst_node_id) in enumerate(zip(src, dst)):  # 遍历每个edge的两个端点
            src_node_label = -1  # 起始node的标签
            dst_node_label = -1  # 终点node的标签
            for value in frame_obj.values():
                if src_node_id in value['node_id']:  # 起始点
                    index = value['node_id'].index(src_node_id)
                    src_node_label = value['labels'][index]
                    if node_features[src_node_id, :].sum() == 0:
                        node_features[src_node_id, :] = value['features'][index]  # 更新该节点特征信息
                if dst_node_id in value['node_id']:
                    index = value['node_id'].index(dst_node_id)
                    dst_node_label = value['labels'][index]
                    if node_features[dst_node_id, :].sum() == 0:
                        node_features[dst_node_id, :] = value['features'][index]
            if src_node_id != -1 and dst_node_label != -1 and src_node_label == dst_node_label:
                edge_label[edge_indx] = 1  # 将该edge设置为1，表示这两个目标为同一个

        graph.ndata['feature'] = node_features
        graph.edata['label'] = edge_label

        pred = gnn_model(graph, [graph for i in range(gnn_model.n_layers)], node_features)
        loss = criterion(pred, edge_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if i % 100 == 0:
            print(
                f"Epoch: {epoch} \t Iter:[{i}/{len(data_loader)}] \t Loss: {loss.item()}")

    return loss_epoch
