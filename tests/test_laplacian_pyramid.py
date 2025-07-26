import pytest
import torch
from torch.testing import assert_close

# Assuming the function laplacian_pyramid_loss is defined in a module named my_module
from topoloss.losses.laplacian_pyramid import laplacian_pyramid_loss

supported_dtypes = [
    torch.float32,
    # torch.float16,
    # torch.bfloat16
]


@pytest.mark.parametrize("dtype1", supported_dtypes)
@pytest.mark.parametrize("dtype2", supported_dtypes)
@pytest.mark.parametrize("height, width", [(16, 16), (32, 32)])
@pytest.mark.parametrize("factor_w, factor_h", [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
@pytest.mark.parametrize("interpolation", ["bilinear", "nearest"])
def test_laplacian_pyramid_loss_precision(
    dtype1, dtype2, height, width, factor_w, factor_h, interpolation
):
    # Create a sample cortical_sheet tensor
    e = 16  # Example depth
    torch.manual_seed(42)  # Set seed for reproducibility
    cortical_sheet = torch.rand(height, width, e).to(dtype1)

    # Call the function with the given precision
    loss = laplacian_pyramid_loss(
        cortical_sheet,
        factor_w=factor_w,
        factor_h=factor_h,
        interpolation=interpolation,
    )

    # Ensure the loss is finite
    assert torch.isfinite(loss), f"Loss is not finite for dtype {dtype1}"

    # Check type consistency
    assert (
        loss.dtype == dtype1
    ), f"Loss dtype {loss.dtype} does not match input dtype {dtype1}."

    # Compare results to float32 (considered ground truth for higher precision)
    if dtype1 != dtype2:
        float32_sheet = cortical_sheet.to(dtype2)
        expected_loss = laplacian_pyramid_loss(
            cortical_sheet=float32_sheet,
            factor_w=factor_w,
            factor_h=factor_h,
            interpolation=interpolation,
        )
        assert_close(
            loss.to(dtype2),
            expected_loss,
            rtol=1e-3,
            atol=1e-4,
            msg=f"Loss mismatch for dtype1 {dtype1} loss: {loss.to(dtype2)} and dtype2 {dtype2} loss2: {expected_loss}",
        )
