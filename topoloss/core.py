import torch.nn as nn
from .utils.getting_modules import get_layer_by_name
from .cortical_sheet import get_weight_cortical_sheet_linear, get_cortical_sheet_conv
from .losses.laplacian import laplacian_pyramid_loss, LaplacianPyramidLoss
from typing import Union


class TopoLoss:
    def __init__(
        self,
        losses: list[Union[LaplacianPyramidLoss]],
    ):
        self.losses = losses

    def check_layer_type(self, layer):
        assert isinstance(
            layer, Union[nn.Conv2d, nn.Linear]
        ), f"Expect layer to be either nn.Conv2d or nn.Linear, but got: {type(layer)}"

    def compute(self, model, reduce_mean=True):
        layer_wise_losses = {}

        for loss_info in self.losses:
            layer = get_layer_by_name(model=model, layer_name=loss_info.layer_name)
            self.check_layer_type(layer=layer)

            if isinstance(layer, nn.Linear):
                cortical_sheet = get_weight_cortical_sheet_linear(layer=layer)
            else:
                cortical_sheet = get_cortical_sheet_conv(layer=layer)

            if isinstance(loss_info, LaplacianPyramidLoss):
                loss = laplacian_pyramid_loss(
                    cortical_sheet=cortical_sheet,
                    factor_h=loss_info.factor_h,
                    factor_w=loss_info.factor_w,
                )
                if loss_info.scale is not None:
                    loss = loss * loss_info.scale
                else:
                    loss = None

            if loss is not None:
                layer_wise_losses[loss_info.layer_name] = loss
            else:
                ## do not backprop if scale is set to None
                ## scale == None means logging only
                pass

        if reduce_mean:
            loss_values = layer_wise_losses.values()
            return sum(loss_values) / len(loss_values)
        else:
            return layer_wise_losses
