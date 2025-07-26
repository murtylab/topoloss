from torchtyping import TensorType
from einops import rearrange
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Union, Optional
from ..utils.getting_modules import get_name_by_layer
from dataclasses import dataclass, field
import torch


def laplacian_pyramid_loss(
    cortical_sheet: TensorType["height", "width", "e"],
    factor_w: float,
    factor_h: float,
    interpolation: str = "bilinear",
):
    grid = cortical_sheet
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
        grid, scale_factor=(1 / factor_h, 1 / factor_w), mode=interpolation
    )
    # Upscale the downscaled grid tensor
    upscaled_grid = F.interpolate(
        downscaled_grid, size=grid.shape[2:], mode=interpolation
    )

    # Calculate the MSE loss between the original grid and upscaled grid
    # loss = F.mse_loss(upscaled_grid, grid)

    grid = rearrange(grid.squeeze(0), "e h w -> (h w) e")
    upscaled_grid = rearrange(upscaled_grid.squeeze(0), "e h w -> (h w) e")
    loss = 1 - F.cosine_similarity(grid, upscaled_grid, dim=-1).mean()

    return loss


@dataclass
class LaplacianPyramid:
    """
    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    - `shrink_factor`: factor by which the grid is shrinked before it gets resized back to it's original size
    """

    layer_name: str
    factor_h: float
    factor_w: float
    interpolation: str = "bilinear"
    scale: Optional[Union[None, float]] = field(default=1.0)

    @classmethod
    def from_layer(
        cls,
        model,
        layer,
        factor_h,
        factor_w,
        scale=1.0,
        interpolation: str = "bilinear",
    ):
        layer_name = get_name_by_layer(model=model, layer=layer)
        return cls(
            layer_name=layer_name,
            scale=scale,
            factor_h=factor_h,
            factor_w=factor_w,
            interpolation=interpolation,
        )


def laplacian_pyramid_loss_on_bias(
    cortical_sheet: TensorType["h", "w"],
    factor_w: float,
    factor_h: float,
    interpolation: str = "bilinear",
):

    grid = cortical_sheet
    assert grid.ndim == 2, "Expected grid to be a 2d tensor of shape (h, w)"

    assert (
        factor_h <= grid.shape[0]
    ), f"Expected factor_h to be <= grid.shape[1] = {grid.shape[1]} but got: {factor_h}"
    assert (
        factor_w <= grid.shape[1]
    ), f"Expected factor_w to be <= grid.shape[2] = {grid.shape[2]} but got: {factor_w}"

    ## h,w -> 1,1,h,w
    grid = grid.unsqueeze(0).unsqueeze(0)
    # Downscale the grid tensor
    downscaled_grid = F.interpolate(
        grid, scale_factor=(1 / factor_h, 1 / factor_w), mode=interpolation
    )
    # Upscale the downscaled grid tensor
    upscaled_grid = F.interpolate(
        downscaled_grid, size=grid.shape[2:], mode=interpolation
    )

    grid = rearrange(grid.squeeze(0).squeeze(0), "h w -> (h w)").unsqueeze(0)
    upscaled_grid = rearrange(
        upscaled_grid.squeeze(0).squeeze(0), "h w -> (h w)"
    ).unsqueeze(0)

    loss = 1 - F.cosine_similarity(upscaled_grid, grid).mean()

    return loss


@dataclass
class LaplacianPyramidOnBias:
    """
    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    - `shrink_factor`: factor by which the grid is shrinked before it gets resized back to it's original size
    """

    layer_name: str
    factor_h: float
    factor_w: float
    interpolation: str = "bilinear"
    scale: Optional[Union[None, float]] = field(default=1.0)

    @classmethod
    def from_layer(
        cls,
        model,
        layer,
        factor_h,
        factor_w,
        scale=1.0,
        interpolation: str = "bilinear",
    ):
        assert (
            layer.bias is not None
        ), "Expected layer to have a bias, but got None. *sad sad sad*"
        layer_name = get_name_by_layer(model=model, layer=layer)
        return cls(
            layer_name=layer_name,
            scale=scale,
            factor_h=factor_h,
            factor_w=factor_w,
            interpolation=interpolation,
        )


@dataclass
class LaplacianPyramidOnInput:
    """
    LaplacianPyramid makes the output space of the layer topographic
    This loss makes the input space topographic.

    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    - `shrink_factor`: factor by which the grid is shrinked before it gets resized back to it's original size
    """

    layer_name: str
    factor_h: float
    factor_w: float
    interpolation: str = "bilinear"
    scale: Optional[Union[None, float]] = field(default=1.0)

    @classmethod
    def from_layer(
        cls,
        model,
        layer,
        factor_h,
        factor_w,
        scale=1.0,
        interpolation: str = "bilinear",
    ):
        layer_name = get_name_by_layer(model=model, layer=layer)
        return cls(
            layer_name=layer_name,
            scale=scale,
            factor_h=factor_h,
            factor_w=factor_w,
            interpolation=interpolation,
        )
