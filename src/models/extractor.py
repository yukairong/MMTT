import copy

import torch
import torch.nn.functional as F
from torch import nn

from src.models.transformer import TransformerDecoderLayer, TransformerDecoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    :param activation:
    :return:
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class Extractor(nn.Module):

    def __init__(self, deformable=False, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        if deformable:
            raise ValueError("Extractor module don't support deformable decoder...")
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                    normalize_before)
            decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, objs, mask, query_num, pos_embed):
        """
        特征融合层
        :param objs: decoder输出的新obj
        :param mask:
        :param query_num: query的查询个数
        :param pos_embed:  位置编码
        :return:
        """
        assert isinstance(objs, list)
        # 各个视角下出现的新目标进行叠加
        merge_embed = torch.stack(objs, dim=0)  # shape: (num, 1, 256)
        query_embed_dim = merge_embed.shape[-1]  # 得到每个query的维度

        unique_objs = torch.zeros_like(merge_embed)


class ContrastiveClusterExtractor(nn.Module):
    def __init__(self, feature_dim, cluster_num):
        super(ContrastiveClusterExtractor, self).__init__()
        self.feature_dim = feature_dim  # 输出的维度应该与embedding 维度一致
        self.cluster_num = cluster_num  # 训练时为track不同的person Id
        # instance-level MLP
        self.instance_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        # cluster-level MLP
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, h_i, h_j):
        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, h):
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
