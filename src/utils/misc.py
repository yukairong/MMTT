import time
import datetime
from argparse import Namespace
from typing import List, Optional
from collections import defaultdict, deque

import torch
import torchvision
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F


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
    batch[0] = list(*batch[0])
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = list(*batch[1])
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


def inverse_sigmoid(x, eps=1e-5):
    """
    sigmoid反归一化
    :param x:
    :param eps:
    :return:
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)

    return torch.log(x1 / x2)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    计算准确率
    :param output:
    :param target:
    :param topk:
    :return:
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def get_world_size():
    """
    获取GPU总数
    :return:
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, query_mask=None,
                       reduction=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if not reduction:
        return loss

    if query_mask is not None:
        loss = torch.stack([l[m].mean(0) for l, m in zip(loss, query_mask)])
        return loss.sum() / num_boxes

    return loss.mean(1).sum() / num_boxes


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict

class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque
        :return:
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    @property
    def max(self):
        return max(self.deque)

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, print_freq, delimiter="\t", vis=None, debug=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.vis = vis
        self.print_freq = print_freq
        self.debug = debug

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, epoch=None, header=None):
        i = 0
        if header is None:
            header = 'Epoch: [{}]'.format(epoch)

        world_len_iterable = get_world_size() * len(iterable)

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(world_len_iterable))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data_time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

                if self.vis is not None:
                    y_data = [self.meters[legend_name].median
                              for legend_name in self.vis.viz_opts['legend']
                              if legend_name in self.meters]
                    y_data.append(iter_time.median)

                    self.vis.plot(y_data, i * get_world_size() + (epoch - 1) * world_len_iterable)

                # DEBUG
                # if i != 0 and i % self.print_freq == 0:
                if self.debug and i % self.print_freq == 0:
                    break

            i += 1
            end = time.time()

        # if self.vis is not None:
        #     self.vis.reset()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

