import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass
import math


@dataclass
class GridDimensions2D:
    width: int
    height: int


def find_cortical_sheet_size(area: float):
    length = int(math.sqrt(area))  # Starting with a square shape
    while area % length != 0:
        length -= 1

    breadth = area // length

    return GridDimensions2D(width=breadth, height=length)


def get_weight_cortical_sheet_linear(layer: nn.Linear):
    assert isinstance(layer, nn.Linear)
    weight = layer.weight
    num_output_neurons = weight.shape[0]
    assert weight.ndim == 2
    cortical_sheet_size = find_cortical_sheet_size(area=num_output_neurons)

    return rearrange(
        weight,
        "(height width) n_input -> height width n_input",
        height=cortical_sheet_size.height,
        width=cortical_sheet_size.width,
    )


def get_cortical_sheet_conv(layer: nn.Conv2d):
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
