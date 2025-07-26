from topoloss import TopoLoss, LaplacianPyramid, LaplacianPyramidOnBias
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

supported_dtypes = [
    torch.float32,
    # torch.float16,
    # torch.bfloat16, ### tests fail on bfloat16
]


# Define the fixture that provides the num_steps argument
@pytest.mark.parametrize("num_steps", [2, 9])
@pytest.mark.parametrize("hidden_size", [30, 25])
@pytest.mark.parametrize("init_from_layer", [True, False])
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_loss_linear_weight(
    num_steps: int, hidden_size: int, init_from_layer: bool, dtype: torch.dtype
):  # num_steps is now passed by the fixture

    # Define the model
    model = nn.Sequential(
        nn.Linear(30, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 20)  # 0  # 2
    ).to(dtype)
    model.requires_grad_(True)

    if init_from_layer:
        losses = [
            LaplacianPyramid.from_layer(
                model=model, layer=model[0], scale=1.0, factor_h=2.0, factor_w=2.0
            ),
            LaplacianPyramid.from_layer(
                model=model, layer=model[2], scale=1.0, factor_h=2.0, factor_w=2.0
            ),
        ]
    else:
        losses = [
            LaplacianPyramid(layer_name="0", scale=1.0, factor_h=2.0, factor_w=2.0),
            LaplacianPyramid(layer_name="2", scale=1.0, factor_h=2.0, factor_w=2.0),
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


# Define the fixture that provides the num_steps argument
@pytest.mark.parametrize("num_steps", [2, 9])
@pytest.mark.parametrize("hidden_size", [30, 25])
@pytest.mark.parametrize("init_from_layer", [True, False])
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_loss_linear_bias(
    num_steps: int, hidden_size: int, init_from_layer: bool, dtype: torch.dtype
):  # num_steps is now passed by the fixture

    # Define the model
    model = nn.Sequential(
        nn.Linear(30, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 20)  # 0  # 2
    ).to(dtype)
    model.requires_grad_(True)

    if init_from_layer:
        losses = [
            LaplacianPyramidOnBias.from_layer(
                model=model, layer=model[0], scale=1.0, factor_h=2.0, factor_w=2.0
            ),
            LaplacianPyramidOnBias.from_layer(
                model=model, layer=model[2], scale=1.0, factor_h=2.0, factor_w=2.0
            ),
        ]
    else:
        losses = [
            LaplacianPyramidOnBias(
                layer_name="0", scale=1.0, factor_h=2.0, factor_w=2.0
            ),
            LaplacianPyramidOnBias(
                layer_name="2", scale=1.0, factor_h=2.0, factor_w=2.0
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
