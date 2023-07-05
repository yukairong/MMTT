from argparse import Namespace
from typing import List, Optional

import torchvision
# needed due to empty tensor bug in pytorch and torchvision 0.5
from torch import Tensor


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """

    # if input.numel() > 0:
    #     return torch.nn.functional.interpolate(
    #         input, size, scale_factor, mode, align_corners
    #     )
    #
    # output_shape = _output_size(2, input, size, scale_factor)
    # output_shape = list(input.shape[:-2]) + list(output_shape)
    # return _new_empty_tensor(input, output_shape)
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))

    return namespace
