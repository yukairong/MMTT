import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.utils import box_ops
from src.utils.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from models.detr import DETR, PostProcess

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableDETR(DETR):
    """
    This is the Deformable DETR module that performs object detection
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss)

        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding

        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)

        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []

            # 遍历不同尺度的输出特征图
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]   # 特征的输入维度
                # hidden_dim为d_model, embedding的维度
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))

            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim

            # 不同尺度的embedding映射层,将器转换为一致的维度
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            ])

        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # 初始化
        prior_prob = 0.01
        bias_value = -math.log((1-prior_prob) / prior_prob)
        # class_embed 共享回归头 (self.hidden_dim, num_classes+1)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        # bbox_embed: MLP(self.hidden_dim, self.hidden_dim, 4, 3) 初始化最后一层输出层
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # 对输入的映射层做初始化
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers

        if two_stage:
            num_pred += 1

        if with_box_refine:
            # 6 x Linear(256, 1)
            self.class_embed = _get_clones(self.class_embed, num_pred)
            # 6 x MLP(包含3个线性层)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hook implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        """ The forward expects a NestedTensor, which consists of:
           - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        features_all = features
        features = features[-3:]

        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-3:]

        src_list = []
        mask_list = []
        pos_list = []

        frame_features = [prev_features, features]

        if not self.multi_frame_attention:
            frame_features = [features]

        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention and self.multi_frame_encoding:
                pos_list.extend([p[:, frame] for p in pos[-3:]])
            else:
                pos_list.extend(pos[-3:])

            for l, feat in enumerate(frame_feat):
                src, mask = feat.decompose()

                if self.merge_frame_features:
                    prev_src, _ = prev_features[l].decompose()
                    src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                else:
                    # 将原始各个特征进行映射embedding
                    src_list.append(self.input_proj[l](src))

                mask_list.append(mask)
                assert mask is not None

            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        if self.merge_frame_features:
                            src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        else:
                            src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])

                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention and self.multi_frame_encoding:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        # hs: 多层的decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # memory: 特征
        # init_reference: 初始化的参考点归一化中心坐标 [bs, 300, 2]
        # inter_references: decoder学习到的参考点归一化中心坐标 [n_decoder, bs, num_query, 2]
        # enc_outputs_class = enc_outputs_coord_unact = None
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(src_list, mask_list, pos_list, query_embeds, targets)

        outputs_classes = []    # 分类结果
        outputs_coords = []     # 回归结果
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # [bs, 300, 2] -> [bs, 300, 2] 反归一化  reference在定义时已经被sigmoid归一化过
            reference = inverse_sigmoid(reference)
            # 分类头 1个全连接层 [bs, 300, 256] -> [bs, 300, num_classes]
            outputs_class = self.class_embed[lvl](hs[lvl])
            # 回归头 3个全连接层 [bs, 300, 256] -> [bs, 300, 4]  xywh   xy为偏移量
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference  # 偏移量 + 参考点坐标 -> 最终xy坐标
            outputs_coord = tmp.sigmoid()  # xywh -> 归一化
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)  # [n_decoder, bs, num_query, num_classes] = [6, bs, 300, 91]
        outputs_coord = torch.stack(outputs_coords)  # [n_decoder, bs, num_query, coords] = [6, bs, 300, 4]

        out = {"pred_logits": outputs_class[-1],
               "pred_boxes": outputs_coord[-1],
               "hs_embed": hs[-1]}

        # 每层的loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape

        # 将每个src输入的embedding对应的memory部分取出来。memory[list 4]features_all[list 4]对应，hs[6,1,300,256]是每层decoder的输出
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        memory = memory_slices

        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformablePostProcess(PostProcess):
    """
    This module converts the model's output into the format expected by the coco api
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        scores, labels = prob.max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # from relative [0,1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
