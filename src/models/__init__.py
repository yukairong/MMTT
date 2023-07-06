import torch

from models.backbone import build_backbone
from models.matcher import build_matcher

def build_model(args):
    """
    构建网络模型
    :param args: 传入配置文件参数
    :return:
    """
    if args.dataset == "wildTrack":
        num_classes = 1
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    # 构建backbone以及 matcher
    backbone = build_backbone(args)
    matcher = build_matcher(args)

