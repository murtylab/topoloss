import torch.nn as nn
from einops import rearrange
from .common import find_cortical_sheet_size, GridDimensions2D


def get_cortical_sheet_linear_input(layer: nn.Linear, strict_layer_type: bool = True):
    if strict_layer_type is True:
        assert isinstance(layer, nn.Linear)
    weight = layer.weight
    num_input_neurons = weight.shape[1]
    assert weight.ndim == 2
    cortical_sheet_size = find_cortical_sheet_size(area=num_input_neurons)

    return rearrange(
        weight,
        "o (h w) -> h w o",
        h=cortical_sheet_size.height,
        w=cortical_sheet_size.width,
        o=weight.shape[0],
    )
