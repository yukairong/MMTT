import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


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
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    # 单层EncoderLayer
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        """
            搭建一层EncoderLayer
        :param d_model: embed_dimension
        :param nhead: 多头注意力的头数
        :param dim_feedforward: FFN的隐层数
        :param dropout:
        :param activation:
        :param normalize_before: 层归一化在之前还是之后
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """

        :param src: CNN 输出的features
        :param src_mask: CNN输出的features自带的mask
        :param src_key_padding_mask: src哪些部分是padding来的
        :param pos: 位置编码
        :return:
        """
        # q k v 在送入Encoder层之前归一化
        src2 = self.norm1(src)
        # 给 q 和 k 添加位置编码
        q = k = self.with_pos_embed(src2, pos)
        # 计算自注意力
        # attn_mask: True表示对应的attention value 不应该存在 src_mask为None
        # key_padding_mask: 输入特征图那部分是padding，这样该部分就不用计算注意力了
        # q k 是对特征query添加位置编码后的特征，value是直接将特征query层归一化
        src2 = self.self_attn(query=q, key=k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # 残差连接
        src = src + self.dropout1(src2)
        # 归一化
        src2 = self.norm2(src)

        # FeedForwardNet
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 残差连接
        src = src + self.dropout2(src2)

        return src

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # q k v 在送入Encoder层之后归一化
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """

        :param src: 经过CNN-backbone得到的features
        :param src_mask: True表示对应的attention value 不应该存在
        :param src_key_padding_mask: 输入特征图那部分是padding，这样该部分就不用计算注意力了
        :param pos: 位置编码
        :return:
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """
        单层DecoderLayer
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        """

        :param d_model: embed_dimension
        :param nhead: 多头注意力的头数
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param normalize_before:
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """

        :param tgt: tgt经过CNN得到的features
        :param memory: encoder的输出
        :param tgt_mask: CNN输出自带的mask
        :param memory_mask: encoder输出自带的mask
        :param tgt_key_padding_mask: tgt部分中哪些部分是padding来的
        :param memory_key_padding_mask: memory中哪些部分是padding来的
        :param pos: encoder那边添加的位置编码
        :param query_pos: tgt添加的位置编码
        :return:
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        # 计算自注意力部分，所以value是 tgt 的值， q k 也是tgt层归一化变换来的
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 计算多头注意力，key 和 value 是encoder的输出 memory, 但是query是tgt来作为查询
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos,
                                 query_pos)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
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

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten N x C x H x W to HW x N x C
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [24*36,2,256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [864,2,256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [100, 2, 256]
        mask = mask.flatten(1)  # [2,864]

        tgt = torch.zeros_like(query_embed)  # [100,2,256]
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # [864,2,256]
        # [6, 100, 2, 256]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


# ---------------------------------------------------------------------------------------------------------------------
def test(args, samples):
    from models.backbone import build_backbone
    device = torch.device(args.device)
    # 构建backbone以及 matcher
    backbone = build_backbone(args)
    backbone = backbone.cuda()
    transformer = build_transformer(args)
    transformer = transformer.cuda()

    samples = samples.to('cuda:0')

    features, pos = backbone(samples)

    src, mask = features[-1].decompose()
    assert mask is not None

    hidden_dim = 256
    num_queries = 100
    num_classes = 1

    input_proj = nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1).cuda()
    query_embed = nn.Embedding(num_queries, hidden_dim).cuda()
    class_embed = nn.Linear(hidden_dim, num_classes + 1).cuda()
    bbox_embed = nn.Linear(hidden_dim, 4).cuda()

    hs = transformer(input_proj(src), mask, query_embed.weight, pos[-1])[0]

    outputs_class = class_embed(hs)
    outputs_coord = bbox_embed(hs).sigmoid()
