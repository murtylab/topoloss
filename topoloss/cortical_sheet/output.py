import torch.nn as nn
from einops import rearrange
from .common import find_cortical_sheet_size, GridDimensions2D


def get_cortical_sheet_linear(layer: nn.Linear, strict_layer_type: bool):
    if strict_layer_type is True:
        assert isinstance(layer, nn.Linear)
    weight = layer.weight
    num_output_neurons = weight.shape[0]
    assert weight.ndim == 2
    cortical_sheet_size = find_cortical_sheet_size(area=num_output_neurons)

    ## is this the same as rearrange(weight, "(h w) i -> h w i")?
    return weight.reshape(
        cortical_sheet_size.height, cortical_sheet_size.width, weight.shape[1]
    )


def get_cortical_sheet_conv(layer: nn.Conv2d, strict_layer_type: bool):
    if strict_layer_type is True:
        assert isinstance(layer, nn.Conv2d)
    weight = layer.weight
    assert weight.ndim == 4
    num_output_channels = weight.shape[0]
    cortical_sheet_size = find_cortical_sheet_size(area=num_output_channels)

    return rearrange(
        weight,
        "(height width) in_channels kernel_height kernel_width -> height width (in_channels kernel_height kernel_width)",
        height=cortical_sheet_size.height,
        width=cortical_sheet_size.width,
    )
