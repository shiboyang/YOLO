import io
import os
import torch
import logging
import numpy as np

from detectron2.layers import Conv2d


def _load_single_tensor(fp, tensor):
    count = tensor.numel()
    w = np.fromfile(fp, dtype=np.float32, count=count)
    w = torch.from_numpy(w).view_as(tensor)
    tensor.data.copy_(w)
    return count


def _load_single_tensor_to_dict(d, fp, name, tensor):
    count = tensor.numel()
    w = np.fromfile(fp, dtype=np.float32, count=count)
    w = torch.from_numpy(w).view_as(tensor)
    d[name] = w
    return count


def load_darknet_weights(weights_path, module_list):
    """
    Load weights from official `darknet53.weights`.

    Official weights file is a binary file where weights are stored in serial order.
    The weights are stored as floats. Extreme care must be taken while loading weight
    file.Some key points to note:

    The weights belong to only two types of layers, either a batch norm layer or a
    convolutional layer.
    The weights are stored in exactly the same order as in configuration file.
    If the convolution layer contains batch normalization, there we will be no bias
    value for convolution.Only weights value will be there for such convolution layers.
    If the convolution layer does not contain batch normalization there will be both
    bias and weight value.

    The first 5 values are header information:
         1. Major version number
         2. Minor Version Number
         3. Subversion number
         4,5. Images seen by the network (during training)
    """
    assert os.path.isfile(weights_path), f"path not exists: {weights_path}"
    module_list = list(module_list)
    logger = logging.getLogger(__name__)

    with open(weights_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        logger.info(f"load official weights (v{header[0]}.{header[1]}.{header[2]})")
        p = f.tell()
        c = 0
        for module in module_list:
            for m in module.modules():
                if not isinstance(m, Conv2d):
                    continue

                bn = m.norm
                if bn is not None:
                    c += _load_single_tensor(f, bn.bias)
                    c += _load_single_tensor(f, bn.weight)
                    c += _load_single_tensor(f, bn.running_mean)
                    c += _load_single_tensor(f, bn.running_var)
                else:
                    # conv layer bias
                    c += _load_single_tensor(f, m.bias)

                # conv layer weight
                c += _load_single_tensor(f, m.weight)
            # print(f"{c + p/4=}, {f.tell()/4=}")


def load_darknet_weights_to_dict(f, module_list, module_prefix="backbone"):
    """
    Load weights from official `darknet53.weights`.

    Official weights file is a binary file where weights are stored in serial order.
    The weights are stored as floats. Extreme care must be taken while loading weight
    file.Some key points to note:

    The weights belong to only two types of layers, either a batch norm layer or a
    convolutional layer.
    The weights are stored in exactly the same order as in configuration file.
    If the convolution layer contains batch normalization, there we will be no bias
    value for convolution.Only weights value will be there for such convolution layers.
    If the convolution layer does not contain batch normalization there will be both
    bias and weight value.

    The first 5 values are header information:
        1. Major version number
        2. Minor Version Number
        3. Subversion number
        4,5. Images seen by the network (during training)
    """
    assert isinstance(f, io.IOBase)
    module_list = list(module_list)
    logger = logging.getLogger(__name__)

    header = np.fromfile(f, dtype=np.int32, count=5)
    logger.info(f"load official weights (v{header[0]}.{header[1]}.{header[2]})")

    state_dict = {}
    for field, m in module_list:
        if not isinstance(m, Conv2d):
            continue
        if module_prefix:
            field = f"{module_prefix}.{field}"

        bn = m.norm
        if bn is not None:
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.bias", bn.bias)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.weight", bn.weight)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.running_mean", bn.running_mean)
            _load_single_tensor_to_dict(state_dict, f, f"{field}.norm.running_var", bn.running_var)
        else:
            # conv layer bias
            _load_single_tensor_to_dict(state_dict, f, f"{field}.bias", m.bias)

        # conv layer weight
        _load_single_tensor_to_dict(state_dict, f, f"{field}.weight", m.weight)
    return state_dict
