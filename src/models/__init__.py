import torch
# from models.detr import SetCriterion
from models.criterion import SetCriterion
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer
from models.deformable_transformer import build_deformable_transformer

from models.detr import DETR, PostProcess
from models.detr_tracking import DeformableDETRTracking, DETRTracking
from models.deformable_detr import DeformableDETR, DeformablePostProcess

from models.multi_view_deformable_tracking import MultiViewDeformableTrack
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
        "person_num": args.person_num,

        # track part
        "track_query_false_positive_prob": args.track_query_false_positive_prob,
        "track_query_false_negative_prob": args.track_query_false_negative_prob,
        "matcher": matcher,
        "backprop_prev_frame": args.track_backprop_prev_frame,
    }

    if args.deformable:
        transformer = build_deformable_transformer(args)

        mmtt_v1_kwargs["transformer"] = transformer
        mmtt_v1_kwargs["num_feature_levels"] = args.num_feature_levels
        mmtt_v1_kwargs["with_box_refine"] = args.with_box_refine
        mmtt_v1_kwargs["two_stage"] = args.two_stage
        mmtt_v1_kwargs["multi_frame_attention"] = args.multi_frame_attention
        mmtt_v1_kwargs["multi_frame_encoding"] = args.multi_frame_encoding
        mmtt_v1_kwargs["merge_frame_features"] = args.merge_frame_features

        # 多视角
        if args.multi_view:
            # TODO: 构建多视角跟踪模型
            model = MultiViewDeformableTrack(mmtt_v1_kwargs)
        else:
            # 跟踪
            if args.tracking:
                if args.masks:
                    raise ValueError("目前暂时不支持分割")
                else:
                    model = DeformableDETRTracking(mmtt_v1_kwargs)
            # 检测
            else:
                if args.masks:
                    raise ValueError("目前暂时不支持分割")
                else:
                    model = DeformableDETR(**mmtt_v1_kwargs)
    # 普通的transformer
    else:
        transformer = build_transformer(args)

        mmtt_v1_kwargs["transformer"] = transformer

        if args.tracking:
            if args.masks:
                raise ValueError("目前暂时不支持分割")
            else:
                model = DETRTracking(mmtt_v1_kwargs)
        else:
            if args.masks:
                raise ValueError("目前暂时不支持分割")
            else:
                model = DETR(**mmtt_v1_kwargs)

    # 损失函数的构建
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_instance': args.instance_loss_coef,
                   'loss_cluster': args.cluster_loss_coef}

    # 是否使用辅助训练
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'instances', 'clusters']

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
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,
        instance_temperature=args.instance_temperature,
        cluster_temperature=args.cluster_temperature,
        track_ids_length=args.person_num)
    criterion.to(device)

    if args.focal_loss:
        postprocessors = {'bbox': DeformablePostProcess()}
    else:
        postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors


