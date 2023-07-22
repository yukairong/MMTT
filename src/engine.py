import logging
import math
import os
import sys
from typing import Iterable

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
        x_i, x_j = backbone.decoder_forward(samples)

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
