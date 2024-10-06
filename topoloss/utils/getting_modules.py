import torch.nn as nn
from functools import reduce


def get_layer_by_name(model: nn.Module, layer_name: str):
    """Retrieve a pytorch layer from a model by it's name"""
    if layer_name != "":
        names = layer_name.split(sep=".")
    else:
        return model
    return reduce(getattr, names, model)


def get_name_by_layer(model: nn.Module, layer: nn.Module):
    """Retrieve the name of a layer from a PyTorch model"""
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None
