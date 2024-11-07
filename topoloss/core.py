"""
Lets start by reproducing the nesim loss here
"""
import math
from typing import Tuple
from dataclasses import dataclass


@dataclass
class GridDimensions2D:
    width: int
    height: int


def find_rectangle_dimensions(area: int) -> GridDimensions2D:
    """
    Find the length and breadth of a rectangle with the least perimeter for a given area.

    Args:
        area (int): The area of the rectangle.

    Returns:
        GridDimensions2D: contains the length and breadth of the rectangle.
    """
    length = int(math.sqrt(area))  # Starting with a square shape
    while area % length != 0:
        length -= 1

    breadth = area // length

    return GridDimensions2D(width=breadth, height=length)

import torch.nn as nn
from functools import reduce


def get_module_by_name(module: nn.Module, name: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.

    Args:
        module (Union[TensorType, nn.Module]): module whose submodule you want to access
        name (str): the string representation of the submodule. Like `"module.something.this_thing"`

    Returns:
        object: module that you wanted to extract
    """
    if name != "":
        names = name.split(sep=".")
    else:
        return module
    return reduce(getattr, names, module)

import torch.nn as nn
import einops

"""
The base grid is an abstraction that is common to both neighbourhood cossim loss
and the laplacian pyramid loss. It takes a linear or a conv layer and arranges it's weights
in a 3 dimensional grid of shape: (height, width, embedding_size)

embedding_size depends on the nature of the layer i.e input size, output size and the kernel size for conv layers.
height * width should be the same as the output size of the model
"""


class BaseGrid2dLinear:
    def __init__(self, linear_layer, height: int, width: int, device: str):
        """
        Initializes a BaseGrid2dLinear object.

        Args:
            linear_layer (nn.Linear): Linear layer for which the Weight grid is created.
            height (int): Height of the Weight grid.
            width (int): Width of the Weight grid.
            device (str): Device on which the Weight grid should be placed.
        """
        self.width = width
        self.height = height
        self.device = device

        assert isinstance(
            linear_layer, nn.Linear
        ), f"linear_layer expected an instance of torch.nn.Linear but got: {type(linear_layer)}"
        # linear_layer.weight.shape: (output, input)
        assert (
            linear_layer.weight.shape[0] == self.width * self.height
        ), f"Expected grid height * width to be the same as linear_layer.weight.shape[0]: {linear_layer.weight.shape[0]}"

        self.linear_layer = linear_layer

    @property
    def grid(self):
        return self.linear_layer.weight.reshape(
            self.height, self.width, self.linear_layer.weight.shape[1]
        ).to(self.device)


class BaseGrid2dConv:
    def __init__(self, conv_layer, height: int, width: int, device: str):
        """
        Initializes a ConvLayerWeightGrid2D object.

        Args:
            conv_layer (nn.Conv2d): Convolutional layer for which the weight grid is created.
            height (int): Height of the weight grid.
            width (int): Width of the weight grid.
            device (str): Device on which the weight grid should be placed.
        """
        self.width = width
        self.height = height
        self.device = device

        assert isinstance(
            conv_layer, nn.Conv2d
        ), f"conv_layer expected an instance of torch.nn.Conv2d but got: {type(conv_layer)}"
        # conv_layer.weight.shape: output_channels, input_channels, kernel_size, kernel_size
        assert (
            conv_layer.weight.shape[0] == self.width * self.height
        ), f"Expected grid height * width to be the same as conv_weights.shape[0]: {conv_layer.weight.shape[0]}"

        self.conv_layer = conv_layer

    @property
    def grid(self):
        all_embeddings_based_on_weights = einops.rearrange(
            self.conv_layer.weight, "o i h w -> o (i h w)"
        )

        return all_embeddings_based_on_weights.reshape(
            self.height, self.width, all_embeddings_based_on_weights.shape[1]
        ).to(self.device)

import einops
import torch.nn.functional as F
from torchtyping import TensorType

from einops import rearrange


def downscale_upscale_loss(grid: TensorType, factor_w: float, factor_h: float):
    assert grid.ndim == 3, "Expected grid to be a 3d tensor of shape (h, w, e)"
    grid = rearrange(grid, "h w e -> e h w").unsqueeze(0)

    assert (
        factor_h <= grid.shape[1]
    ), f"Expected factor_h to be <= grid.shape[1] = {grid.shape[1]} but got: {factor_h}"
    assert (
        factor_w <= grid.shape[2]
    ), f"Expected factor_w to be <= grid.shape[2] = {grid.shape[2]} but got: {factor_w}"
    # Downscale the grid tensor
    downscaled_grid = F.interpolate(
        grid, scale_factor=(1 / factor_h, 1 / factor_w), mode="bilinear"
    )
    # Upscale the downscaled grid tensor
    upscaled_grid = F.interpolate(downscaled_grid, size=grid.shape[2:], mode="bilinear")

    # Calculate the MSE loss between the original grid and upscaled grid
    # loss = F.mse_loss(upscaled_grid, grid)

    grid = rearrange(grid.squeeze(0), "e h w -> (h w) e")
    upscaled_grid = rearrange(upscaled_grid.squeeze(0), "e h w -> (h w) e")
    loss = 1 - F.cosine_similarity(grid, upscaled_grid, dim=-1).mean()

    return loss    


from typing import List
import torch.nn as nn
from typing import Union


class LaplacianPyramidLoss:
    def __init__(
        self,
        layer: Union[nn.Conv2d, nn.Linear],
        device: str,
        factor_w: List[float] = [2.0],
        factor_h: List[float] = [2.0],
    ):

        output_size = layer.weight.shape[0]
        grid_size = find_rectangle_dimensions(area=output_size)

        if isinstance(layer, nn.Linear):
            
            self.grid_container = BaseGrid2dLinear(
                linear_layer=layer,
                height=grid_size.height,
                width=grid_size.width,
                device=device,
            )

        elif isinstance(layer, nn.Conv2d):

            self.grid_container = BaseGrid2dConv(
                conv_layer=layer,
                height=grid_size.height,
                width=grid_size.width,
                device=device
            )
        else:
            raise TypeError(
                f"Expected layer to be one of nn.Linear or nn.Conv2d but got: {type(layer)}"
            )

        self.factor_w = factor_w
        self.factor_h = factor_h
        self.device = device

    def get_loss(self):
        losses = []
        for factor_h, factor_w in zip(self.factor_w, self.factor_h):

            loss = downscale_upscale_loss(
                grid=self.grid_container.grid, factor_w=factor_w, factor_h=factor_h
            )
            losses.append(loss)

        return sum(losses)
    
import torch.nn as nn
from torchtyping import TensorType
from typing import Union
import wandb

import json
from typing import List, Union
from pydantic import BaseModel




from pydantic import BaseModel, Extra, Field
from typing import Union, List

class NeighbourhoodCosineSimilarity(BaseModel, extra=Extra.forbid):
    """
    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    """

    layer_name: str
    scale: Union[None, float]
    ## loss_type holds the name, makes the json easier to read
    loss_type: str = Field("neighbourhood_cossim", Literal=True, type=str)


class LaplacianPyramid(BaseModel, extra=Extra.forbid):
    """
    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    - `shrink_factor`: factor by which the grid is shrinked before it gets resized back to it's original size
    """

    layer_name: str
    scale: Union[None, float]
    shrink_factor: List[float]
    ## loss_type holds the name, makes the json easier to read
    loss_type: str = Field("laplacian_pyramid", Literal=True, type=str)


class TopoLossConfig(BaseModel, extra="allow"):
    """
    layer_wise_configs: list of SingleLayerConfig instances.
    They specify which layers to watch and apply loss upon.
    """

    layer_wise_configs: List[
        Union[LaplacianPyramid]
    ]

    def save_json(self, filename: str):
        with open(filename, "w") as file:
            json.dump(self.model_dump(), file, indent=4)

    @classmethod
    def from_json(cls, filename: str):
        with open(filename, "r") as file:
            json_data = json.load(file)
        # return cls.model_validate(json_data)
        return cls(**json_data)

## unified API for a single layer loss
class SingleLayerLossHandler:
    def __init__(
        self,
        single_layer_config: Union[
            NeighbourhoodCosineSimilarity, LaplacianPyramid
        ],
        model: nn.Module,
        device: str,
    ):
        layer = get_module_by_name(module=model, name=single_layer_config.layer_name)

        if isinstance(single_layer_config, LaplacianPyramid):
            self.layer_loss = LaplacianPyramidLoss(
                layer=layer,
                device=device,
                factor_h=single_layer_config.shrink_factor,
                factor_w=single_layer_config.shrink_factor,
            )
        else:
            raise TypeError(f"Invalid single_layer_config: {single_layer_config}")

        self.config = single_layer_config

        """
        if scale is not None:
            wandb log the latest loss
        if scale is None:
            compute and then log the loss
        """
        if self.config.scale is not None:
            self.latest_loss = None

        self.scale = self.config.scale

    def compute(self):
        if self.config.scale is not None:
            loss = self.layer_loss.get_loss()
            self.latest_loss = loss.item()

            if isinstance(self.scale, float):
                # raise AssertionError('I am here', self.scale, loss)
                return self.scale * loss
            else:
                # print(f'scale: {self.scale.get_value(step = False)}')
                return self.scale.get_value(step=True) * loss
        else:
            return None

    def __repr__(self):
        return f"SingleLayerLossHandler with loss: {self.layer_loss} on layer: {self.config.layer_name}"

    def get_log_data(self):
        data = {}
        if self.config.scale is not None:
            assert (
                self.latest_loss is not None
            ), "Cannot wandb log the loss before its computed at least once. Run self.compute()"
            data[self.config.layer_name] = self.latest_loss
        else:
            data[self.config.layer_name] = self.layer_loss.get_loss().item()

        return data


class TopoLoss:
    def __init__(self, model: nn.Module, config: TopoLossConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

        self.layer_handlers = []

        for single_layer_config in self.config.layer_wise_configs:
            handler = SingleLayerLossHandler(
                single_layer_config=single_layer_config,
                device=self.device,
                model=self.model,
            )
            self.layer_handlers.append(handler)

    def __repr__(self):
        message = f"TopoLoss: {self.layer_handlers}"
        return message

    def compute(self, reduce_mean=True) -> Union[TensorType, dict]:
        losses_from_each_layer = {}

        for loss_handler in self.layer_handlers:
            loss = loss_handler.compute()

            if loss is not None:
                losses_from_each_layer[loss_handler.config.layer_name] = loss
            else:
                pass

        if reduce_mean is True:
            if len(losses_from_each_layer) > 0:
                return sum(losses_from_each_layer.values()) / len(
                    losses_from_each_layer
                )
            else:
                return None
        else:
            return losses_from_each_layer
        
    def get_log_data(self) -> dict:
        all_data = {}

        for loss_handler in self.layer_handlers:
            if isinstance(loss_handler.layer_loss, LaplacianPyramidLoss):
                prefix = "laplacian_loss_"

            all_data[
                prefix + loss_handler.config.layer_name
            ] = loss_handler.get_log_data()[loss_handler.config.layer_name]

        return all_data

    def wandb_log(self):
        all_data = {}

        all_neighbourhood_cossim_losses = []
        all_dynamic_loss_scales = []

        for loss_handler in self.layer_handlers:
            if isinstance(loss_handler.layer_loss, LaplacianPyramidLoss):
                prefix = "laplacian_loss_"

                if not isinstance(loss_handler.scale, float) is not None:
                    all_dynamic_loss_scales.append(
                        loss_handler.scale.get_value(step=False)
                    )

            all_data[
                prefix + loss_handler.config.layer_name
            ] = loss_handler.get_log_data()

        if len(all_neighbourhood_cossim_losses) > 0:
            wandb.log(
                {
                    "mean_neighbourhood_cosine_similarity": sum(
                        all_neighbourhood_cossim_losses
                    )
                    / len(all_neighbourhood_cossim_losses)
                }
            )

        if len(all_dynamic_loss_scales) > 0:
            wandb.log(
                {
                    f"mean_loss_scale": sum(all_dynamic_loss_scales)
                    / len(all_dynamic_loss_scales)
                }
            )

        wandb.log(all_data)

    def get_all_grid_states(self):
        all_data = {}
        for loss_handler in self.layer_handlers:
            if isinstance(loss_handler.layer_loss, LaplacianPyramidLoss):
                prefix = "laplacian_loss_"

            all_data[
                prefix + loss_handler.config.layer_name
            ] = loss_handler.layer_loss.grid_container.grid

        return all_data