from topoloss import TopoLoss, LaplacianPyramid
import pytest
import torch.nn as nn
import torch.optim as optim
import torch

supported_dtypes = [
    torch.float32,
    # torch.float16,
    # torch.bfloat16,
]


# Define the fixture that provides the num_steps argument
@pytest.mark.parametrize("num_steps", [2, 9])
@pytest.mark.parametrize("hidden_channels", [16, 32])
@pytest.mark.parametrize("init_from_layer", [True, False])
@pytest.mark.parametrize("dtype", supported_dtypes)
@pytest.mark.parametrize("interpolation", ["bilinear", "nearest"])
def test_loss_conv(
    num_steps: int,
    hidden_channels: int,
    init_from_layer: bool,
    dtype,
    interpolation: str,
):  # num_steps is now passed by the fixture

    # Define the model
    model = nn.Sequential(
        nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),  # Conv layer 0
        nn.ReLU(),
        nn.Conv2d(hidden_channels, 12, kernel_size=3, padding=1),  # Conv layer 2
    ).to(dtype=dtype)
    model.requires_grad_(True)

    if init_from_layer:
        losses = [
            LaplacianPyramid.from_layer(
                model=model,
                layer=model[0],
                scale=1.0,
                factor_h=3.0,
                factor_w=3.0,
                interpolation=interpolation,
            ),
            LaplacianPyramid.from_layer(
                model=model,
                layer=model[2],
                scale=1.0,
                factor_h=3.0,
                factor_w=3.0,
                interpolation=interpolation,
            ),
        ]
    else:
        losses = [
            LaplacianPyramid(
                layer_name="0",
                scale=1.0,
                factor_h=3.0,
                factor_w=3.0,
                interpolation=interpolation,
            ),
            LaplacianPyramid(
                layer_name="2",
                scale=1.0,
                factor_h=3.0,
                factor_w=3.0,
                interpolation=interpolation,
            ),
        ]

    # Define the TopoLoss
    tl = TopoLoss(
        losses=losses,
    )

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    losses = []

    # Training loop
    for step_idx in range(num_steps):
        loss = tl.compute(reduce_mean=True, model=model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients after each step
        losses.append(loss.item())  # Use .item() to get the scalar value

    # Assertion to verify loss decreases
    assert (
        losses[-1] < losses[0]
    ), f"Expected loss to go down for {num_steps} training steps for dtype: {dtype}, but it did not. \x1B[3msad sad sad\x1B[23m"
