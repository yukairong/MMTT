import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding
from src.utils.misc import NestedTensor, is_main_process

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # backbone非训练状态,或是layer2,layer3,layer4不在backbone中, 将参数锁定
            if (not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # 获取想要的中间层输出
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm"""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):

        norm_layer = FrozenBatchNorm2d
        # 使用getattr()方法获取torchvision.models模块中的函数,函数名称由变量name来指定
        backbone = getattr(torchvision.models, name)(
            # 用于指定是否在模型中使用dilation来替换stride。某个元素为True，则表示将使用dilation来代替对应层中的stride
            replace_stride_with_dilation=[False, False, dilation],
            # is_main_process()用于指示当前进程是否为主进程，如果当前进程是主进程，则会加载预训练模型，否则不会加载
            pretrained=is_main_process(), norm_layer=norm_layer
        )
        super().__init__(backbone, train_backbone, return_interm_layers)

        if dilation:
            # 感受野扩大一倍,步长相应缩减一半,确保输出一致
            self.strides[-1] = self.strides[-1] // 2





def build_backbone(args):
    """
    构建backbone
    :param args: 配置参数
    :return:
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)  # 是否取出中间几层特征图

    backbone = Backbone(args.backbone,
                        train_backbone,
                        return_interm_layers,
                        args.dilation)

    print(backbone)