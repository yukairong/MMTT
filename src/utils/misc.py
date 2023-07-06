from argparse import Namespace
from typing import List, Optional

import torch
import torchvision
from torch import Tensor
import torch.distributed as dist


# *********************************************************************************************************************
def nested_dict_to_namespace(dictionary):
    """
    将args的字典形式转化为Namespace格式
    :param dictionary:
    :return:
    """
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))

    return namespace


# *********************************************************************************************************************
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor

     Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this class can go away.

    :param input:
    :param size:
    :param scale_factor:
    :param mode:
    :param align_corners:
    :return:
    """

    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


# *********************************************************************************************************************
def collate_fn(batch):
    """
    将一个batch的数据重新组装为自定义的形式，输入参数batch就是原始的一个batch数据，
    通常在Pytorch中的Dataloader中，会将一个batch的数据组装为((data1, label1), (data2, label2), ...)这样的形式，
    于是第一行代码的作用就是将其变为[(data1, data2, data3, ...), (label1, label2, label3, ...)]这样的形式，
    然后取出batch[0]即一个batch的图像输入到 nested_tensor_from_tensor_list() 方法中进行处理，
    最后将返回结果替代原始的这一个batch图像数据
    :param batch: 原始的一个batch数据
    :return:
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])

    return tuple(batch)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    首先，为了能够统一batch中所有图像的尺寸，以便形成一个batch，我们需要得到其中的最大尺度（在所有维度上），
    然后对尺度较小的图像进行填充（padding），同时设置mask以指示哪些部分是padding得来的，
    以便后续模型能够在有效区域内去学习目标，相当于加入了一部分先验知识
    :param tensor_list:
    :return:
    """

    if tensor_list[0].ndim == 3:
        # 得到一个batch中所有图像张量每个维度的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 指示图像中哪些位置是padding的部分
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # 原始图像有效部分位置设为false，padding部分设置为true
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    """
    何得到batch中每张图像在每个维度上的最大值
    :param the_list:
    :return:
    """
    #  (List[List[int]]) -> List[int]
    maxes = the_list[0]

    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    return maxes


class NestedTensor(object):
    """
        将 tensor 和 mask 打包
    """

    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """
            将数据转换到指定的设备上
        :param device:
        :return:
        """
        # (Device) -> NestedTensor
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None

        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """
        将数据tensor和掩码mask分离
        :return:
        """
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def unmasked_tensor(self, index: int):
        tensor = self.tensors[index]
        # 数据本身就没有mask，直接返回
        if not self.mask[index].any():
            return tensor

        h_index = self.mask[index, 0, :].nonzero(as_tuple=True)[0]
        if len(h_index):
            tensor = tensor[:, :, :h_index[0]]

        w_index = self.mask[index, :, 0].nonzero(as_tuple=True)[0]
        if len(w_index):
            tensor = tensor[:, :w_index[0], :]

        return tensor

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


# *********************************************************************************************************************
def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)

        return output

    return dictionary.to(device)
