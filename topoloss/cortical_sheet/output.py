import torch.nn as nn
from einops import rearrange
from .common import find_cortical_sheet_size, GridDimensions2D


def get_cortical_sheet_linear(layer: nn.Linear, strict_layer_type: bool, custom_weight_attribute_name = None):
    if strict_layer_type is True:
        assert isinstance(layer, nn.Linear)
    if custom_weight_attribute_name is not None:
        assert hasattr(layer, custom_weight_attribute_name), f"Expected layer to have attribute {custom_weight_attribute_name}, but it does not. Available attributes are: {dir(layer)}"
        weight = getattr(layer, custom_weight_attribute_name)
    else:
        weight = layer.weight
    num_output_neurons = weight.shape[0]
    assert weight.ndim == 2, f"Expected weight to be a 2d tensor of shape (out_features, in_features), but got weight of shape: {weight.shape}"
    cortical_sheet_size = find_cortical_sheet_size(area=num_output_neurons)

    ## is this the same as rearrange(weight, "(h w) i -> h w i")?
    return weight.reshape(
        cortical_sheet_size.height, cortical_sheet_size.width, weight.shape[1]
    )


def get_cortical_sheet_conv(layer: nn.Conv2d, strict_layer_type: bool, custom_weight_attribute_name = None):
    if strict_layer_type is True:
        assert isinstance(layer, nn.Conv2d)
    if custom_weight_attribute_name is not None:
        assert hasattr(layer, custom_weight_attribute_name), f"Expected layer to have attribute {custom_weight_attribute_name}, but it does not. Available attributes are: {dir(layer)}"
        weight = getattr(layer, custom_weight_attribute_name)
    else:
        weight = layer.weight
    assert weight.ndim == 4, f"Expected weight to be a 4d tensor of shape (out_channels, in_channels, kernel_height, kernel_width), but got weight of shape: {weight.shape}"
    num_output_channels = weight.shape[0]
    cortical_sheet_size = find_cortical_sheet_size(area=num_output_channels)

    return rearrange(
        weight,
        "(height width) in_channels kernel_height kernel_width -> height width (in_channels kernel_height kernel_width)",
        height=cortical_sheet_size.height,
        width=cortical_sheet_size.width,
    )
