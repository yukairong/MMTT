import copy

import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc import (NestedTensor, accuracy, dice_loss, get_world_size,
                        interpolate, is_dist_avail_and_initialized,
                        nested_tensor_from_tensor_list, sigmoid_focal_loss)

class DETR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, overflow_boxes=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.num_queries = num_queries  # 300
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes    # False
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """
        Returns the hidden feature dimension size.
        :return:
        """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """
        Returns FPN channels
        :return:
        """
        return self.backbone.num_channels[:3][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        src = self.input_proj(src)
        pos = pos[-1]

        batch_size, _, _, _ = src.shape

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and "track_query_hs_embeds" in targets[0]:
            # [BATCH_SIZE, NUM_PROBS, 4]
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])
            num_track_queries = track_query_hs_embeds.shape[1]
            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim
            ).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed
            ], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

            for i, target in enumerate(targets):
                target["track_query_hs_embeds"] = tgt[:, i]

        assert mask is not None
        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, pos, tgt
        )

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "hs_embed": hs_without_norm[-1]
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord
            )

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api
    """
    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # from relative [0, 1] to absoluate [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """

        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = self.process_boxes(out_bbox, target_sizes)

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'scores_no_object': s_n_o}
            for s, l, b, s_n_o in zip(scores, labels, boxes, prob[..., -1])
        ]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x