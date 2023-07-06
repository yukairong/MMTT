import math
import torch
from torch import nn
from utils.misc import NestedTensor

class PositionEmbeddingSine3D(nn.Module):

    def __init__(self, num_pos_feats=64, num_frames=2, temperature=10000, normalize=False, scale=None):
        """

        :param num_pos_feats: 位置编码特征维度
        :param num_frames: 帧数量
        :param temperature:
        :param normalize: 是否标准化
        :param scale: 放缩尺寸
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.frames = num_frames

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        n, h, w = mask.shape

        mask = mask.view(n, 1, h, w)
        mask = mask.expand(n, self.frames, h, w)

        assert mask is not None
        not_mask = ~mask

        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6

            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)

        return pos


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    绝对位置embedding, 可学习
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos


def build_position_encoding(args):
    """
    构建位置编码
    :param args:
    :return:
    """
    position_embedding = None

    if args.multi_frame_attention and args.multi_frame_encoding:
        n_steps = args.hidden_dim // 3
        sine_emedding_func = PositionEmbeddingSine3D
    else:
        n_steps = args.hidden_dim // 2
        sine_emedding_func = PositionEmbeddingSine

    # 使用sine方式进行位置编码
    if args.position_embedding == "sine":
        position_embedding = sine_emedding_func(n_steps,  normalize=True)
    # 使用学习的方式进行位置编码
    elif args.position_embedding == "learned":
        position_embedding = PositionEmbeddingLearned(n_steps)

    return position_embedding


if __name__ == '__main__':
    import sacred
    from utils.misc import nested_dict_to_namespace

    ex = sacred.Experiment("position_encoding_experiment")
    @ex.config
    def pos_encode_test_config():
        multi_frame_attention = False
        multi_frame_encoding = True
        hidden_dim = 256
        position_embedding = "sine"
        device = "cuda:0"

    config = pos_encode_test_config()
    ex.add_config(config)

    args = nested_dict_to_namespace(config)

    position_embedding = build_position_encoding(args)

    tensors = torch.randn(size=(1, 3, 40, 40), dtype=torch.float32, device=args.device)
    masks = torch.randint(0, 1, size=(1, 40, 40), dtype=torch.int8, device=args.device)
    test_input = NestedTensor(tensors, masks)
    output = position_embedding(test_input)
    print(output)

