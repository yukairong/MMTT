import math
import random
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from src.utils.misc import NestedTensor, nested_tensor_from_tensor_list

from models.matcher import HungarianMatcher
from models.deformable_detr import DeformableDETR

class MultiViewDeformableTrack(nn.Module):

    def __init__(self, kwargs):
        super().__init__()
        self.deformable_detr = DeformableDETR(**kwargs)
        self._matcher = kwargs["matcher"]
        self._track_query_false_positive_prob = kwargs["track_query_false_positive_prob"]
        self._track_query_false_negative_prob = kwargs["track_query_false_negative_prob"]
        self._backprop_prev_frame = kwargs["backprop_prev_frame"]

        self._tracking = False
        self.track_pools = {}   # 存储每个track person的特征信息
        self.frame_new_obj_hs = []  # 存储t帧下各个视角认为可能的新对象

    def train(self, mode: bool = True):
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        self.eval()
        self._tracking = True

    def forward(self, samples: NestedTensor, targets: list=None, prev_features=None):

        if targets is not None and not self._tracking:
            # 取出batch_size中所有的t-1帧图像
            prev_targets = [target['prev_target'] for target in targets]

            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if "prev_prev_image" in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target["prev_target"] = target["prev_prev_target"]

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # TODO: 单路多帧
                    else:
                        # prev_out: t-1帧下的存储信息, 存储pred_logits, pred_boxes, hs_embed, aux_outputs
                        # prev_features: t-1帧下的backbone上获得的features
                        # prev_memory: t-1帧下的embedding对应的decoder输出, [(batch_size, channels, height, width)]
                        # prev_hs: t-1帧下的多层的decoder输出 [n_decoder, bs, num_query, d_model]
                        prev_out, _, prev_features, prev_memory, prev_hs = self.deformable_detr([t["prev_image"] for t in targets]) # t-1帧计算

                    # 对prev_out, prev_features, prev_memory以及prev_hs进行处理
                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.item() if "aux_outputs" not in k
                    }

                    # t-1帧的decoder输出和标注 pre_outputs与pre_targets进行匹配
                    prev_indices = self._matcher(prev_outputs_without_aux,
                                                 prev_targets)

                    device = prev_out["pred_boxes"].device
                    # prev_indices: [34, 17, 25]
                    # 当前batch的图片中prev_image对应的最小目标数, 即可能存在共有的目标数(0, min{34, 17, 25})
                    min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])

                    num_prev_target_ind = 0
                    # 只要有目标存在就随机挑选一个最小公共目标值
                    if min_prev_target_ind:
                        num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

                    # 假阳的数量
                    num_prev_target_ind_for_fps = 0
                    if num_prev_target_ind:
                        # [0, 3)
                        num_prev_target_ind_for_fps = torch.randint(
                            # (假阳概率 * 公共的目标数) + 1
                            int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)
                        ).item()

                    for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
                        # prev_out_ind: decoder输出的300个query中,作为输出目标对应的id数 [0, 300)
                        # prev_target_ind: 匈牙利匹配得到的 标注文件对应的track_id的值索引
                        prev_out_ind, prev_target_ind = prev_ind

                        # random subset
                        if self._track_query_false_negative_prob:
                            # 在第t帧输入track_query时,对其按照一定比率进行抹除,以消除对t-1帧信息的依赖,这个操作可以减少对track_query的依赖,
                            # 从而保证检测和跟踪的平衡性
                            # 对之前的帧图像中的目标做掩码(模拟遮挡), 来减少对t-1帧的依赖
                            random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]
                            prev_out_ind = prev_out_ind[random_subset_mask] # t-1帧的输出匹配索引
                            prev_target_ind = prev_target_ind[random_subset_mask]   # t-1帧的掩码后真实匹配索引

                        # detected prev frame tracks
                        # t-1帧时刻所有已经做过掩码操作后的track ID
                        prev_track_ids = target["prev_target"]["track_ids"][prev_target_ind]

                        # match track ids between frames
                        # 同一视角下帧间匹配,寻找当前图像中与前一帧图像track_id相同的
                        # target_ind_match_matrix:
                        #   row - 当前图片的所有目标数量
                        #   col - t-1帧图像的所有目标数量 (len_col < len_row)
                        #   每一列最多只有一个为True
                        # target_ind_matching: t-1帧中的目标是否在当前帧出现
                        # target_ind_matched_idx: t帧匹配的目标是t-1帧对应的哪个目标track_id(关联)
                        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                        target_ind_matching = target_ind_match_matrix.any(dim=1)
                        target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                        target["track_query_match_ids"] = target_ind_matched_idx    # t帧和t-1帧匹配的track ID索引

                        # TODO：取出matched的特征,并把track query部分放入历史存储池中
                        matched_idx_list = prev_out_ind[target_ind_matching]  # 匹配到的对应的track id在预测输出的序号列表
                        # 如果对应的匹配结果不为空,那么将对应的
                        if len(matched_idx_list) > 0:
                            for index_in_list, output_index in enumerate(matched_idx_list):
                                track_id = target_ind_matched_idx[index_in_list]
                                prev_hs_matched = prev_hs[-1, i, output_index, :]

                                # 将该track id的特征值存入track pool中
                                self.track_pools[track_id].append(prev_hs_matched)

                        # random false positive
                        # if add_false_pos: 始终为True
                        prev_out_ind = prev_out_ind.to(device)
                        # t-1帧图像中的目标在t帧也存在的对应预测框
                        prev_boxes_matched = prev_out["pred_boxes"][i, prev_out_ind[target_ind_matching]]

                        # not_prev_out_ind: 新添加进来的目标
                        not_prev_out_ind = torch.arange(prev_out["pred_boxes"].shape[1])
                        not_prev_out_ind = [ind.item() for ind in not_prev_out_ind if ind not in prev_out_ind]

                        random_false_out_ind = []
                        # num_prev_target_ind: 当前batch图片对应的前一帧图像中公共的目标数量
                        # prev_target_ind_for_fps: 公共的目标取随机个?
                        prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                        for j in prev_target_ind_for_fps:
                            # TODO：将认为可能是新目标的特征存储起来
                            multi_view_new_obj_hs = prev_hs[-1, i, j, :]
                            self.frame_new_obj_hs.append(multi_view_new_obj_hs) # 将其放入frame_new_obj_hs中,以备后续融合

                            # prev_boxes_unmatched: 当前batch中第i张图片未能与t-1帧目标匹配的预测框
                            prev_boxes_unmatched = prev_out["pred_boxes"][i, not_prev_out_ind]

                            if len(prev_boxes_matched) > j: # 判断j是否没有超出prev_boxes_matched范围
                                prev_box_matched = prev_boxes_matched[j]
                                # 匹配到的中心值 - 未匹配到的box中心值
                                box_weights = prev_box_matched.unsqueeze(dim=0)[:, :2] - prev_boxes_unmatched[:, :2]
                                box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                                box_weights = torch.sqrt(box_weights)   # 计算L2损失
                                # 挑出一个最类似于背景的query作为目标(数据增强) (偏离目标中心点最远的query)
                                random_false_out_idx = not_prev_out_ind.pop(torch.multinomial(box_weights.cpu(), 1).item())
                            else:
                                random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])
                            # 保存把背景的query作为有目标的query(索引)
                            random_false_out_ind.append(random_false_out_idx)
                        # 已经检测出来的目标 + 把背景query作为有目标的query 索引
                        prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                        # 实际上有目标的query是真的有目标,还是数据增强得到的有目标的query (用于区分prev_out_ind)
                        target_ind_matching = torch.cat([
                            target_ind_matching,
                            torch.tensor([False,] * len(random_false_out_ind)).bool().to(device)
                        ])

                    # track query masks
                    track_queries_mask = torch.ones_like(target_ind_matching).bool()
                    track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
                    track_queries_fal_pos_mask[~target_ind_matching] = True

                    # TODO: 将其放入多视角目标融合模块

                    # set prev frame info
                    target['multi_track_query_hs_embeds'] = None    # 存储多视角跟踪特征信息
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
            else:
                pass

        out, targets, features, memory, hs = self.deformable_detr(samples, targets, prev_features)

        return out, targets, features, memory, hs
















