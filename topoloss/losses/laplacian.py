from einops import rearrange
import torch.nn.functional as F
from torchtyping import TensorType
from pydantic import BaseModel, Field
from typing import Union, Optional
from ..utils.getting_modules import get_name_by_layer


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


def laplacian_pyramid_loss(
    cortical_sheet: TensorType["height", "width", "depth"], factor_w: float, factor_h
):
    assert (
        cortical_sheet.ndim == 3
    ), f"Expected cortical_sheet to have 3 dims, but got: {cortical_sheet.ndim}"
    cortical_sheet = rearrange(cortical_sheet, "h w e -> e h w").unsqueeze(0)

    assert (
        factor_h <= cortical_sheet.shape[2]
    ), f"Expected factor_h ({factor_h}) to be <= cortical_sheet.shape[1] ({cortical_sheet.shape[2]}). For reference, cortical_sheet.shape: {cortical_sheet.shape}"
    assert (
        factor_w <= cortical_sheet.shape[3]
    ), f"Expected factor_w ({factor_w}) to be <= cortical_sheet.shape[2] ({cortical_sheet.shape[3]}). For reference, cortical_sheet.shape: {cortical_sheet.shape}"
    # Downscale the cortical_sheet tensor
    downscaled_cortical_sheet = F.interpolate(
        cortical_sheet, scale_factor=(1 / factor_h, 1 / factor_w), mode="bilinear"
    )
    # Upscale the downscaled cortical_sheet tensor
    upscaled_cortical_sheet = F.interpolate(
        downscaled_cortical_sheet, size=cortical_sheet.shape[2:], mode="bilinear"
    )

    cortical_sheet = rearrange(cortical_sheet.squeeze(0), "e h w -> (h w) e")
    upscaled_cortical_sheet = rearrange(
        upscaled_cortical_sheet.squeeze(0), "e h w -> (h w) e"
    )
    loss = (
        1 - F.cosine_similarity(cortical_sheet, upscaled_cortical_sheet, dim=-1).mean()
    )

    return loss
