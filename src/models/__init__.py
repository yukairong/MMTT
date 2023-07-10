import torch
from models.detr import SetCriterion
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer
from models.deformable_transformer import build_deformable_transformer


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

    transformer = build_transformer(args)
    if args.deformable:
        transformer = build_deformable_transformer(args)
        print("Deformable Transformer Model: ")
        print(transformer)

    # 损失函数的构建
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef, }

    # 是否使用辅助训练
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight, )
    criterion.to(device)


