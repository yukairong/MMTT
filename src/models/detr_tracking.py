import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn

from utils import box_ops
from utils.misc import NestedTensor, get_rank
from models.deformable_detr import DeformableDETR
from models.detr import DETR
from models.matcher import HungarianMatcher

class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame = False,
                 **kwargs):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame

        self._tracking = False

    def train(self, mode: bool = True):
        """
        Sets the module in train model
        :param mode:
        :return:
        """
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """
        Sets the module in tracking mode.
        :return:
        """
        self.eval()
        self._tracking = True

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        device = prev_out["pred_boxes"].device

        # prev_indices: [34, 17，25] 当前batch图片的prev_image对应的最小目标数
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])
        # 找出当前batch图片的prev_image共有的目标数 (0, min{34, 17, 25}]
        num_prev_target_ind = 0
        if min_prev_target_ind:
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

        num_prev_target_ind_for_fps = 0
        if num_prev_target_ind:
            # [0,3)
            num_prev_target_ind_for_fps = torch.randint(
                # (假阳概率 * 公共的目标数) + 1
                int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)
            ).item()

        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            # prev_out_ind: decoder 输出的300个query中，作为输出目标对应的ID数 【0， 300）
            # prev_target_ind: 匈牙利匹配得出的 标注文件对应的"track_ID"值索引
            prev_out_ind, prev_target_ind = prev_ind

            # random subset
            if self._track_query_false_negative_prob:
                # 在第二帧输入track query时，对其按一定比率进行抹除，以消除对前一帧的依赖。这个操作可以减少对track query的依赖，
                # 从而保证检测和跟踪的平衡性。
                # 对之前的帧图像中的目标做掩码（模拟遮挡），来减少对上一帧的依赖
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]
                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]

            # detected prev frame tracks
            # 当前图像的前一帧图像中的所有目标取其中筛选过的track Id
            prev_track_ids = target["prev_target"]["track_ids"][prev_target_ind]

            # match track ids between frames
            # 帧间匹配,寻找当前图像中与前一帧图像track_ID相同的
            # target_ind_match_matrix：行 - --- 当前图片的所有目标数量
            #                          列 - --- 前一帧图像的所有目标数量 所以【列数 < 行数】
            #                          每一列最多只有一个为True 【以上分析未考虑新目标加入？】
            # target_ind_matching: 前一帧图像中的目标是否在当前帧出现
            # target_ind_matched_idx：当前帧匹配的目标是前一帧对应的哪个目标track_ID（关联）
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            target['track_query_match_ids'] = target_ind_matched_idx # 当前帧与上一帧匹配的track ID索引

            # random false positive
            if add_false_pos:
                prev_out_ind = prev_out_ind.to(device)
                # prev_out: Decoder输出的token再经过FFN后的结果
                # prev_out_ind: decoder输出的300个query中,作为输出目标对应的ID数量 [0, 300]
                # target_ind_matching: 前一帧图像中的目标是否在当前帧出现,然后根据pre_target_ind取对应的pre_track_ids,
                #           与真正的track_ids作比较,得到target_ind_matching（True, True, True, True, True) 和 track_ids的下标
                # Return
                #   prev_boxes_matched: 前一帧图像中的目标在当前帧也存在对应的预测框
                prev_boxes_matched = prev_out["pred_boxes"][i, prev_out_ind[target_ind_matching]]

                # not_prev_out_ind: 理解为新添加进来的目标
                not_prev_out_ind = torch.arange(prev_out["pred_boxes"].shape[1])
                not_prev_out_ind = [ind.item() for ind in not_prev_out_ind if ind not in prev_out_ind]

                random_false_out_ind = []
                # num_prev_target_ind: 当前batch图片对应的前一帧图像中公共的目标数量
                # prev_target_ind_for_fps: 公共的目标取随机个?
                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                for j in prev_target_ind_for_fps:
                    # prev_boxes_unmatched: 当前batch中第i张图片中未能与上一帧目标匹配的预测框
                    prev_boxes_unmatched = prev_out["pred_boxes"][i, not_prev_out_ind]

                    if len(prev_boxes_matched) > j: # 判断j是否没有超出prev_boxes_matched范围
                        prev_box_matched = prev_boxes_matched[j]
                        # 匹配到的中心值 - 未匹配到的box中心值
                        box_weights = prev_box_matched.unsqueeze(dim=0)[:, :2] - prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)   # 计算L2损失
                        # 挑出一个最类似于背景的query作为目标 (数据增强) (偏离目标中心点最远的query)
                        random_false_out_idx = not_prev_out_ind.pop(torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])
                    # 保存把背景的query作为有目标的query (索引)
                    random_false_out_ind.append(random_false_out_idx)
                # 已经检测出来的目标 + 把背景query作为有目标的query 索引
                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                # 实际上有目标的query是真的有目标,还是数据增强得到的有目标的query (用于区分prev_out_ind)
                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target["track_query_boxes"] = prev_out["pred_boxes"][i, prev_out_ind].detach()

            target["track_queries_mask"] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target["track_queries_fal_pos_mask"] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if 'prev_prev_image' in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target["prev_target"] = target["prev_prev_target"]

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # PREV PREV
                        prev_prev_out, _, prev_prev_features, _, _ = super().forward(
                            [t['prev_prev_image'] for t in targets])

                        prev_prev_outputs_without_aux = {
                            k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                        prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                        self.add_track_queries_to_targets(
                            prev_targets, prev_prev_indices, prev_prev_out, add_false_pos=False)

                        # PREV
                        prev_out, _, prev_features, _, _ = super().forward(
                            [t['prev_image'] for t in targets],
                            prev_targets,
                            prev_prev_features)
                    else:
                        prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets]) # t-1帧计算

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux,
                                                 prev_targets)  # t-1帧的decoder输出和标注  pre_outputs与pre_targets匹配

                    self.add_track_queries_to_targets(targets, prev_indices, prev_out)  # pre_targets和targets匹配
            else:
                # if not training we do not add track queries and evaluate detection performance only
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target["track_query_hs_embeds"] = torch.zeros(0, self.hidden_dim).float().to(device)
                    target["track_queries_mask"] = torch.zeros(self.num_queries).bool().to(device)
                    target["track_query_boxes"] = torch.zeros(0, 4).to(device)
                    target["track_query_match_ids"] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs = super().forward(samples, targets, prev_features)

        return out, targets, features, memory, hs

class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, kwargs):
        DETR.__init__(self, **kwargs)
        DETRTrackingBase.__init__(self, **kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, kwargs):
        DeformableDETR.__init__(self, **kwargs)
        DETRTrackingBase.__init__(self, **kwargs)