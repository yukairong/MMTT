import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        创建Matcher匹配器

        :param cost_class: 分类误差在匹配成本中的相对权重
        :param cost_bbox: 匹配成本中边界框坐标的L1误差的相对权重
        :param cost_giou: 匹配成本中边界框的giou损失的相对权重
        :param focal_loss:  是否使用focal loss
        :param focal_alpha: focal loss的alpha值
        :param focal_gamma: focal loss的gamma值
        """
        super().__init__()
        self.cost_class = cost_class    # 2.0
        self.cost_bbox = cost_bbox  # 5.0
        self.cost_giou = cost_giou  # 2.0
        self.focal_loss = focal_loss    # True
        self.focal_alpha = focal_alpha  # 0.25
        self.focal_gamma = focal_gamma  # 2

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        计算match匹配情况
        :param outputs: 一个字典包含至少包含下面这些信息
            "pred_logits": 形状为 [batch_size, num_queries, num_classes], 分类的预测值
            "pred_boxes": 形状为 [batch_size, num_queries, 4], 预测的框坐标值
        :param targets: 一个目标的列表 (len(targets) = batch_size), 每个target是一个字典，应该会包含下面两个参数
            "label": 形状为 [num_target_boxes], 包含每个目标框的标签
            "boxes": 形状为 [num_target_boxes, 4], 包含每个目标框的坐标
        :return:
            一个形状为 batch_size, 包含元组(index_i, index_j)的列表
                - index_i 代表所选预测的目标索引
                - index_j 代表对应所选目标的索引
            对应每个batch的元素,它都需要保证
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]  # 取出输出的batch_size和num_queries

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]

        # 每个样本的每个queries的各个类别的置信度
        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        # 将坐标(cx, cy, w, h) -> (x1, y1, x2, y2), 计算GIOU
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
                + self.cost_class * cost_class \
                + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # 送入训练的每张图片各有多少目标个数
        sizes = [len(v["boxes"]) for v in targets]

        # cost_matrix: [target_sample, pred_queries, ground_truth_box]
        for i, target in enumerate(targets):
            if "track_query_match_ids" not in target:
                continue

            prop_i = 0
            # 每个queries取出,进行匹配
            for j in range(cost_matrix.shape[1]):
                if target['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost_matrix[i, j] = np.inf  # FP 第i张图下的j query判断错误,所以全部设置为inf
                elif target["track_queries_mask"][j]:
                    track_query_id = target["track_query_match_ids"][prop_i]
                    prop_i += 1

                    cost_matrix[i, j] = np.inf
                    cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
                    cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )