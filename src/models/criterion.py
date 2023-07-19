import copy

import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc import (NestedTensor, accuracy, dice_loss, get_world_size,
                        interpolate, is_dist_avail_and_initialized,
                        nested_tensor_from_tensor_list, sigmoid_focal_loss)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    """ Create the criterion.
           Parameters:
               num_classes: number of object categories, omitting the special no-object category
               matcher: module able to compute a matching between targets and proposals
               weight_dict: dict containing as key the names of the losses and as values their
                            relative weight.
               eos_coef: relative classification weight applied to the no-object category
               losses: list of all the losses to be applied. See get_loss for list of
                       available losses.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, tracking, track_query_false_positive_eos_weight):
        """

        :param num_classes: 这里不考虑 no-object category 【背景类】
        :param matcher: targets 和 预测值 之间的匹配【匈牙利】
        :param weight_dict: 权重字典，有18个 key
            loss_ce = 2, loss_bbox = 5, loss_giou = 2，还有对应的 loss_ce; loss_bbox; loss_giou_{0-4}
        :param eos_coef: 0.1 背景类的权重 no-object category
        :param losses: 需要计算损失值列表 ['labels', 'boxes', 'cardinality']
        :param focal_loss: True 是否使用 focal_loss
        :param focal_alpha: focal_loss的参数 0.25
        :param focal_gamma: focal_loss的参数 2
        :param tracking: True
        :param track_query_false_positive_eos_weight: True
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict  # dict: 18  3x6  6个decoder的损失权重   6 * (loss_ce + loss_giou + loss_bbox)
        self.eos_coef = eos_coef  # 0.1
        self.losses = losses  # list: 3  ['labels', 'boxes', 'cardinality']
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # 背景类的权重 0.1
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tracking = tracking
        self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight

    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        outputs：'pred_logits'=[bs, 100, 92] 'pred_boxes'=[bs, 100, 4] 'aux_outputs'=5 * ([bs, 100, 92]+[bs, 100, 4])
        targets：'boxes'=[3,4] labels=[3] ...
        indices： [3] 如：5,35,63  匹配好的3个预测框idx

        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 分类：[bs, 100, 类别数]

        # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片
        # 1=[num_all_gt] 记录每个匹配到的预测框的index
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 正样本+负样本  上面匹配到的预测框作为正样本 正常的idx  而100个中没有匹配到的预测框作为负样本(idx=91 背景类)
        target_classes[idx] = target_classes_o

        # 分类损失 = 正样本 + 负样本; 正样本+负样本=100，正样本个数=GT个数，负样本个数=100-GT个数；
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        if self.tracking and self.track_query_false_positive_eos_weight:
            for i, target in enumerate(targets):
                if 'track_query_boxes' in target:
                    # remove no-object weighting for false track_queries
                    loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                    # assign false track_queries to some object class for the final weighting
                    target_classes = target_classes.clone()
                    target_classes[i, target['track_queries_fal_pos_mask']] = 0

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        num_boxes：当前batch的所有gt个数
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']  # [300, 2, 1]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.

            outputs：'pred_logits'=[bs, 100, 92] 'pred_boxes'=[bs, 100, 4] 'aux_outputs'=5*([bs, 100, 92]+[bs, 100, 4])
            targets：'boxes'=[3,4] labels=[3] ...
            indices： [3] 如：5,35,63  匹配好的3个预测框idx
            num_boxes：当前batch的所有gt个数
        """
        assert 'pred_boxes' in outputs
        # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
        idx = self._get_src_permutation_idx(indices)
        # [all_gt_num, 4]  这个batch的所有正样本的预测框坐标
        src_boxes = outputs['pred_boxes'][idx]
        # [all_gt_num, 4]  这个batch的所有gt框坐标
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 计算GIOU损失
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # 'loss_bbox': L1回归损失   'loss_giou': giou回归损失
        # 回归损失：只计算所有正样本的回归损失；
        # 回归损失 = L1 Loss + GIOU Loss

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                有4个key：pred_logits: [1, 329, 1] track_query + object_query
                         pred_boxes: [1, 329, 4]
                         hs_embed: [1. 329. 256]
                         aux_outputs: list 5,每个list 是一个字典，每个字典里有 pred_boxes 和  pred_logits
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc

            dict: 'pred_logits'=Tensor[bs, 100, 92个class]  'pred_boxes'=Tensor[bs, 100, 4]  最后一个decoder层输出
                             'aux_output'={list:5}  0-4  每个都是dict:2 pred_logits+pred_boxes 表示5个decoder前面层的输出

             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      每张图片包含以下信息：'boxes'、'labels'、'image_id'、'area'、'iscrowd'、'orig_size'、'size'
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # dict: 2   最后一个decoder层输出 : pred_logits[bs, 100, 7个class] + pred_boxes[bs, 100, 4]
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 匈牙利算法  解决二分图匹配问题  从100个预测框中找到和N个gt框一一对应的预测框  其他的100-N个都变为背景
        # Retrieve the matching between the outputs of the last layer and the targets  list:1
        # tuple: 2    0=Tensor3=Tensor[5, 35, 63]  匹配到的3个预测框  其他的97个预测框都是背景
        #             1=Tensor3=Tensor[1, 0, 2]    对应的三个gt框
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)  # int 统计这整个batch的所有图片的gt总个数  3
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():  # False
            torch.distributed.all_reduce(num_boxes)
        # get_world_size 获取GPU的数量
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # self.losses：list3:[labels, boxes, cardinality]
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # two_stage 才会用到的
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses