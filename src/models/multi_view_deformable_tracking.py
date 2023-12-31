import copy
import math
from contextlib import nullcontext

import torch
from torch import nn

from src.models.deformable_detr import DeformableDETR
from src.models.graphSAGE import GraphSAGE
from src.utils.misc import NestedTensor


class MultiViewDeformableTrack(nn.Module):

    def __init__(self, kwargs):
        super().__init__()
        self.deformable_detr = DeformableDETR(**kwargs)
        # self.extractor = ContrastiveClusterExtractor(self.deformable_detr.hidden_dim, kwargs["person_num"])
        self.extractor = GraphSAGE(in_feats=self.deformable_detr.hidden_dim, n_hidden=kwargs['gnn_hidden_feats'],
                                   out_feats=kwargs['gnn_out_feats'], n_classes=kwargs['gnn_edge_classes'],
                                   n_layers=kwargs['gnn_n_layers'], activation=kwargs['gnn_activation'],
                                   dropout=kwargs['gnn_dropout'], aggregator=kwargs['gnn_aggregator'])
        # 融合模块
        self.merge_module = nn.Conv2d(self.deformable_detr.hidden_dim * 2, self.deformable_detr.hidden_dim,
                                      kernel_size=1)
        self._matcher = kwargs["matcher"]
        self._track_query_false_positive_prob = kwargs["track_query_false_positive_prob"]
        self._track_query_false_negative_prob = kwargs["track_query_false_negative_prob"]
        self._backprop_prev_frame = kwargs["backprop_prev_frame"]

        self._tracking = False
        self.track_pools = {}  # 存储每个track person的特征信息
        self.frame_new_obj_hs = []  # 存储t帧下各个视角认为可能的新对象
        self.frame_new_obj_hs_augment = []  # 存储t帧下其他decoder可能新对象的特征

        self.frame_obj = {}  # 存储t帧下每个视角的对象特征以及真实track id

    def train(self, mode: bool = True):
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        self.eval()
        self._tracking = True

    def _frame_reset(self):
        self.frame_new_obj_hs = []
        self.frame_new_obj_hs_augment = []
        self.frame_obj = {}

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        # 清空frame_new_obj特征
        self._frame_reset()

        if targets is not None and not self._tracking:
            # 取出batch_size中所有的t-1帧图像
            prev_targets = [target['prev_target'] for target in targets]

            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    # TODO: 单路多帧训练部分
                    if False:
                        # if "prev_prev_image" in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target["prev_target"] = target["prev_prev_target"]

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                    else:
                        # prev_out: t-1帧下的存储信息, 存储pred_logits, pred_boxes, hs_embed, aux_outputs
                        # prev_features: t-1帧下的backbone上获得的features
                        # prev_memory: t-1帧下的backbone对应的输出做embeding, [(batch_size, channels, height, width)]
                        # prev_hs: t-1帧下的多层的decoder输出 [n_decoder, bs, num_query, d_model]
                        prev_out, _, prev_features, prev_memory, prev_hs = self.deformable_detr(
                            [t["prev_image"] for t in targets])  # t-1帧计算

                    # 对prev_out, prev_features, prev_memory以及prev_hs进行处理
                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if "aux_outputs" not in k
                    }

                    # t-1帧的decoder输出和标注 pre_outputs与pre_targets进行匹配， 300 和 4 做一个匹配
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
                            # (假阳概率 * 公共的目标数) + 1，确保【假阳】目标 少于 【真阳】目标数。否则【阳】中 更多的是假阳，不利于收敛
                            int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)
                        ).item()

                    prev_track_ids_list = []
                    for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):  # 考虑batch，循环{batch}次
                        # prev_out_ind: decoder输出的300个query中,作为输出目标对应的id数 [0, 300)
                        # prev_target_ind: 匈牙利匹配得到的 标注文件对应的track_id的值索引
                        prev_out_ind, prev_target_ind = prev_ind

                        # 将目标放入frame_obj中
                        for out_ind, target_ind in zip(prev_out_ind, prev_target_ind):
                            obj_feat = prev_hs[-1, i, out_ind, :]  # 目标的特征
                            obj_label = target['prev_target']['track_ids'][target_ind]  # 目标的track id

                            if i not in self.frame_obj:
                                self.frame_obj[i] = {
                                    'features': [],
                                    'labels': []
                                }
                            self.frame_obj[i]['features'].append(obj_feat)
                            self.frame_obj[i]['labels'].append(obj_label)

                        src_prev_out_ind = copy.deepcopy(prev_out_ind)
                        # random subset
                        if self._track_query_false_negative_prob:
                            # 在第t帧输入track_query时,对其按照一定比率进行抹除,以消除对t-1帧信息的依赖,这个操作可以减少对track_query的依赖,
                            # 从而保证检测和跟踪的平衡性
                            # 对之前的帧图像中的目标做掩码(模拟遮挡), 来减少对t-1帧的依赖
                            random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]
                            prev_out_ind = prev_out_ind[random_subset_mask]  # t-1帧的输出匹配索引 [14 221]  221
                            prev_target_ind = prev_target_ind[random_subset_mask]  # t-1帧的掩码后真实匹配索引

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

                        target["track_query_match_ids"] = target_ind_matched_idx  # t帧和t-1帧匹配的track ID索引

                        # 取出matched的特征,并把track query部分放入历史存储池中
                        # prev_out_ind = prev_out_ind.cpu()
                        matched_idx_list = prev_out_ind[target_ind_matching.cpu()]  # 匹配到的对应的track id在预测输出的序号列表
                        new_obj_ind_matching = [i for i in src_prev_out_ind if
                                                i not in matched_idx_list]  # 获取新匹配的query id
                        new_obj_matched_idx_list = new_obj_ind_matching
                        # new_obj_matched_idx_list = prev_out_ind[new_obj_ind_matching]   # 匹配到的新的object id再预测输出的序号列表

                        # 如果对应的新obj结果不为空,那么将对应的object id的特征值存入
                        # 将认为可能是新目标的特征存储起来
                        # if len(new_obj_matched_idx_list) > 0:
                        # for output_index in new_obj_matched_idx_list:
                        #     # 第6层decoder输出特征
                        #     multi_view_new_obj_hs = prev_hs[-1, i, output_index, :]
                        #     self.frame_new_obj_hs.append(multi_view_new_obj_hs)  # 将其放入frame_new_obj_hs中,以备后续融合
                        #
                        #     # 第5层decoder输出特征
                        #     multi_view_new_obj_hs_augment = prev_hs[-2, i, output_index, :]
                        #     self.frame_new_obj_hs_augment.append(multi_view_new_obj_hs_augment)

                        # if len(src_prev_out_ind) > 0:
                        #     for output_index in src_prev_out_ind:
                        #         multi_view_new_obj_hs = prev_hs[-1, i, output_index, :]
                        #         self.frame_new_obj_hs.append(multi_view_new_obj_hs)
                        #
                        #         multi_view_new_obj_hs_augment = prev_hs[-2, i, output_index, :]
                        #         self.frame_new_obj_hs_augment.append(multi_view_new_obj_hs_augment)
                        # else:
                        #     multi_view_new_obj_hs = prev_hs[-1, i, 0, :]
                        #     self.frame_new_obj_hs.append(multi_view_new_obj_hs)
                        #
                        #     multi_view_new_obj_hs_augment = prev_hs[-2, i, 0, :]
                        #     self.frame_new_obj_hs_augment.append(multi_view_new_obj_hs_augment)

                        batch_track_ids_list = []
                        # 如果对应的匹配结果不为空,那么将对应的track id的特征值存入track pool中
                        if len(matched_idx_list) > 0:
                            for index_in_list, output_index in enumerate(prev_out_ind):
                                target_track_id_index = prev_target_ind[index_in_list]
                                track_id = target["prev_target"]["track_ids"][target_track_id_index]
                                batch_track_ids_list.append(track_id)
                                prev_hs_matched = prev_hs[-1, i, output_index, :]

                                # 将该track id的特征值存入track pool中
                                if track_id.cpu() not in self.track_pools:
                                    self.track_pools[int(track_id.cpu())] = []
                                self.track_pools[int(track_id.cpu())].append(prev_hs_matched.cpu())

                        prev_track_ids_list.append(batch_track_ids_list)
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
                        # prev_target_ind_for_fps: 公共的目标取随机个? ********************************
                        prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                        for j in prev_target_ind_for_fps:  # 就 j 是编号【index】
                            # prev_boxes_unmatched: 当前batch中第i张图片未能与t-1帧目标匹配的预测框
                            prev_boxes_unmatched = prev_out["pred_boxes"][i, not_prev_out_ind]

                            if len(prev_boxes_matched) > j:  # 判断j是否没有超出prev_boxes_matched范围
                                prev_box_matched = prev_boxes_matched[j]
                                # 匹配到的中心值 - 未匹配到的box中心值
                                box_weights = prev_box_matched.unsqueeze(dim=0)[:, :2] - prev_boxes_unmatched[:, :2]
                                box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                                box_weights = torch.sqrt(box_weights)  # 计算L2损失
                                # 挑出一个最类似于背景的query作为目标(数据增强) (偏离目标中心点最远的query)
                                random_false_out_idx = not_prev_out_ind.pop(
                                    torch.multinomial(box_weights.cpu(), 1).item())
                            else:
                                random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])
                            # 保存把背景的query作为有目标的query(索引)
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
                        # target['multi_track_query_hs_embeds'] = None  # 存储多视角跟踪特征信息
                        target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
                        target["track_query_boxes"] = prev_out["pred_boxes"][i, prev_out_ind].detach()

                        target["track_queries_mask"] = torch.cat([
                            track_queries_mask,
                            torch.tensor([False, ] * self.deformable_detr.num_queries).to(device)
                        ]).bool()

                        target["track_queries_fal_pos_mask"] = torch.cat([
                            track_queries_fal_pos_mask,
                            torch.tensor([False, ] * self.deformable_detr.num_queries).to(device)
                        ]).bool()

            else:
                # if not training we do not add track queries and evaluate detection performance only
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target["track_query_hs_embeds"] = torch.zeros(0, self.deformable_detr.hidden_dim).float().to(device)
                    target["track_queries_mask"] = torch.zeros(self.deformable_detr.num_queries).bool().to(device)
                    target["track_query_boxes"] = torch.zeros(0, 4).to(device)
                    target["track_query_match_ids"] = torch.tensor([]).long().to(device)

        # TODO: 将新目标送入提取模块中
        # new_obj_features = torch.stack(self.frame_new_obj_hs)
        # new_obj_features_augment = torch.stack(self.frame_new_obj_hs_augment)
        # pred_instance_i, pred_instance_j, pred_cluster_i, pred_cluster_j = \
        #     self.extractor(new_obj_features, new_obj_features_augment)
        #
        # pred_cluster_ids = self.extractor.forward_cluster(new_obj_features)

        # extractor_features = {}
        # 将所有为一类的特征放在一起
        # for index, cluster_id in enumerate(pred_cluster_ids):
        #     feature = self.frame_new_obj_hs[index]  # 取出当前新目标的特征
        #     if cluster_id not in extractor_features:
        #         extractor_features[cluster_id] = [feature]
        #         continue
        #     extractor_features[cluster_id].append(feature)  # 合并同类项

        # multi_track_query_embeds = []  # 存储各个视角下全部的目标信息
        # if len(extractor_features) > 0:  # 存在新的对象
        #     for obj_index, features in extractor_features.items():
        #         num = len(features)
        #         merge_feature = features[0]
        #         # 只要features不只有一个，就进行融合
        #         while num != 1:
        #             tmp = torch.hstack(features[:2])[None, :, None, None].to(device)
        #             merge_feature = self.merge_module(tmp)
        #             # 将两个特征向量融合为1个 [0 1 2 3] - [6]
        #             features.insert(2, merge_feature)
        #             features = features[2:]
        #
        #             num -= 1

        # multi_track_query_embeds.append(merge_feature)  # 融合后的特征放入多视角目标信息中

        if len(self.track_pools) > 0:  # 对历史跟踪池信息进行融合
            for track_index, track_features in self.track_pools.items():
                num = len(track_features)
                merge_feature = track_features[0]
                # 只要features不只有一个，就进行融合
                while num != 1:
                    tmp = torch.hstack(track_features[:2])[None, :, None, None].to(device)
                    merge_feature = self.merge_module(torch.cat(track_features[:2], dim=1))
                    # 将两个特征向量融合为1个
                    track_features.insert(2, merge_feature)
                    track_features = track_features[2:]
                    num -= 1

                self.track_pools[track_index] = [merge_feature]

                # multi_track_query_embeds.append(merge_feature)
        for batch_id in range(len(prev_track_ids_list)):  # 取该batch内的第几张图片
            for step, batch_track_id in enumerate(prev_track_ids_list[batch_id]):
                if batch_track_id in self.track_pools:
                    targets[batch_id]["track_query_hs_embeds"][step] = self.track_pools[batch_track_id][0]

        # 更新所有的target信息,将跟踪目标信息加入其中
        # for target in targets:
        #     target["track_query_hs_embeds"] = [embed.to(device) for embed in multi_track_query_embeds]

        out, targets, features, memory, hs = self.deformable_detr(samples, targets, prev_features)

        # 将Extractor的返回值也加入out中
        # out.update(
        #     {"pred_instance_i": pred_instance_i,
        #      "pred_instance_j": pred_instance_j,
        #      "pred_cluster_i": pred_cluster_i,
        #      "pred_cluster_j": pred_cluster_j}
        # )

        return out, targets, features, memory, hs

    def decoder_forward(self, samples: NestedTensor, targets: list = None):
        with torch.no_grad():
            # 获取t-1帧的所有图像
            prev_targets = [target['prev_target'] for target in targets]
            # 获取t-2帧的所有图像
            for target, prev_target in zip(targets, prev_targets):
                prev_target['prev_target'] = target['prev_prev_target']
            prev_prev_targets = [target['prev_prev_target'] for target in targets]

            # 获取t-2帧的model输出
            prev_prev_out, _, prev_prev_features, prev_prev_memory, prev_prev_hs = self.deformable_detr(
                [t['prev_prev_image'] for t in targets]
            )
            prev_prev_outputs_without_aux = {
                k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k
            }
            # t-2帧下queries和实际t-2帧下的obj进行匹配
            prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

            # 获取t-1帧的model输出
            prev_out, _, prev_features, prev_memory, prev_hs = self.deformable_detr(
                [t['prev_image'] for t in targets]
            )
            prev_outputs_without_aux = {
                k: v for k, v in prev_out.items() if 'aux_outputs' not in k
            }
            # t-1帧下queries和实际t-1帧下的obj进行匹配
            prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

            device = prev_out['pred_boxes'].device  # 获取device

            obj_hs_dict = {}
            # t-2帧进行遍历
            for i, (target, prev_prev_ind) in enumerate(zip(targets, prev_prev_indices)):
                prev_prev_out_ind, prev_prev_target_ind = prev_prev_ind  # 取出第t-2帧下的匹配成功的queries索引以及对应的track id索引
                # t-2帧时刻下的track id
                prev_prev_track_ids = target['prev_prev_target']['track_ids']
                for out_ind, target_ind in zip(prev_prev_out_ind, prev_prev_target_ind):
                    obj_label = prev_prev_track_ids[target_ind].item()  # 真实的obj标签
                    obj_hs = prev_prev_hs[-1, i, out_ind, :]  # decoder的queries特征

                    if obj_label not in obj_hs_dict:
                        obj_hs_dict[obj_label] = [obj_hs]
                        continue
                    obj_hs_dict[obj_label].append(obj_hs)

            # t-1帧进行遍历
            for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
                prev_out_ind, prev_target_ind = prev_ind
                # t-1帧时刻下的track id
                prev_track_ids = target['prev_target']['track_ids']
                for out_ind, target_ind in zip(prev_out_ind, prev_target_ind):
                    obj_label = prev_track_ids[target_ind].item()
                    obj_hs = prev_hs[-1, i, out_ind, :]

                    if obj_label not in obj_hs_dict:
                        obj_hs_dict[obj_label] = [obj_hs]
                        continue
                    obj_hs_dict[obj_label].append(obj_hs)

            x_i = []
            x_j = []
            for obj_id, obj_hs_list in obj_hs_dict.items():
                length_obj_hs_list = len(obj_hs_list)
                if length_obj_hs_list >= 2:
                    choose_obj_hs_index_list = torch.randperm(length_obj_hs_list)[:2]
                    same_obj_hs_list = torch.stack(obj_hs_list)[choose_obj_hs_index_list]

                    x_i.append(same_obj_hs_list[0])
                    x_j.append(same_obj_hs_list[1])

            obj_features_i = torch.stack(x_i)
            obj_features_j = torch.stack(x_j)

            return obj_features_i, obj_features_j

    def inference(self, samples: NestedTensor, targets: list = None, prev_features=None):
        # if targets is None:
        #     # prev_targets = [target['prev_target'] for target in targets]
        #     for target in targets:
        #         device = 'cuda'
        #
        #         target['track_query_hs_embeds'] = torch.zeros(0, self.deformable_detr.hidden_dim).float().to(device)
        #         # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
        #         target['track_queries_mask'] = torch.zeros(self.deformable_detr.num_queries).bool().to(device)
        #         target['track_queries_fal_pos_mask'] = torch.zeros(self.deformable_detr.num_queries).bool().to(device)
        #         target['track_query_boxes'] = torch.zeros(0, 4).to(device)
        #         target['track_query_match_ids'] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs = self.deformable_detr(samples, targets, prev_features)

        return out, targets, features, memory, hs
