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

    # version 1
    mmtt_v1_kwargs = {
        "backbone": backbone,
        "num_classes": num_classes - 1 if args.focal_loss else num_classes,
        "num_queries": args.num_queries,
        "aux_loss": args.aux_loss,
        "overflow_boxes": args.overflow_boxes,

        # track part
        "track_query_false_positive_prob": args.track_query_false_positive_prob,
        "track_query_false_negative_prob": args.track_query_false_negative_prob,
        "matcher": matcher,
        "backprop_prev_frame": args.track_backprop_prev_frame,
    }

    # TODO: build deformable transformer
    if args.deformable:
        transformer = None


