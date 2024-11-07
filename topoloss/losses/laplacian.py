from einops import rearrange
import torch.nn.functional as F
from torchtyping import TensorType
from pydantic import BaseModel, Field
from typing import Union, Optional
from ..utils.getting_modules import get_name_by_layer
import torch.nn.functional as F
from torchtyping import TensorType
from einops import rearrange

class LaplacianPyramidLoss(BaseModel, extra="forbid"):
    """
    - `layer_name`: name of layer in model, something like "model.fc1"
    - `scale`: scale by which the loss for this layer is to be multiplied. If None, then will just watch the layer's loss.
    - `shrink_factor`: factor by which the grid is shrinked before it gets resized back to it's original size
    """

    layer_name: str
    factor_h: float
    factor_w: float
    scale: Optional[Union[None, float]] = Field(default=1.0)

    @classmethod
    def from_layer(cls, model, layer, factor_h, factor_w, scale=1.0):
        layer_name = get_name_by_layer(model=model, layer=layer)
        return cls(
            layer_name=layer_name,
            scale=scale,
            factor_h=factor_h,
            factor_w=factor_w,
        )


def laplacian_pyramid_loss(cortical_sheet: TensorType, factor_w: float, factor_h: float):
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