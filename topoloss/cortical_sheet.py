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

def get_weight_cortical_sheet_linear(
    layer: nn.Linear
):
    assert isinstance(layer, nn.Linear)
    weight = layer.weight.data
    num_output_neurons = weight.shape[0]
    assert weight.ndim == 2
    cortical_sheet_size = find_cortical_sheet_size(
        area=num_output_neurons
    )

    return rearrange(
        weight,
        "n_output n_input -> height width n_input",
        height = cortical_sheet_size.height,
        width = cortical_sheet_size.width 
    )
