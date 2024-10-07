import torch.nn as nn
import torch.optim as optim
from topoloss import TopoLoss, LaplacianPyramidLoss
import pytest

# Define the fixture that provides the num_steps argument
@pytest.mark.parametrize("num_steps", [2, 9])
@pytest.mark.parametrize("hidden_size", [30, 25])
@pytest.mark.parametrize("init_from_layer", [True, False])
def test_loss_linear(
    num_steps: int, hidden_size: int, init_from_layer: bool
):  # num_steps is now passed by the fixture

    model = nn.Sequential(
        nn.Linear(30, hidden_size), nn.ReLU(), nn.Linear(30, 20)  # 0  # 2
    )
    model.requires_grad_(True)

    if init_from_layer:
        tl = TopoLoss(
            losses=[
                LaplacianPyramidLoss.from_layer(
                    model=model, layer=model[0], scale=1.0, factor_h=3.0, factor_w=3.0
                ),
                LaplacianPyramidLoss.from_layer(
                    model=model, layer=model[2], scale=1.0, factor_h=3.0, factor_w=3.0
                ),
            ]
        )
    else:

        tl = TopoLoss(
            losses=[
                LaplacianPyramidLoss(
                    layer_name="0", scale=1.0, factor_h=3.0, factor_w=3.0
                ),
                LaplacianPyramidLoss(
                    layer_name="2", scale=1.0, factor_h=3.0, factor_w=3.0
                ),
            ]
        )
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    losses = []
    for step_idx in range(num_steps):
        loss = tl.compute(model=model, reduce_mean=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Make sure to reset gradients after each step
        losses.append(loss.item())  # Use .item() to get the scalar value

    assert (
        losses[-1] < losses[0]
    ), f"Expected loss to go down for {num_steps} training steps, but it did not. \x1B[3msad sad sad\x1B[23m"
