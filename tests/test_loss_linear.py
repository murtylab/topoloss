from topoloss import TopoLoss, LaplacianPyramid
import pytest
import torch.nn as nn
import torch.optim as optim


# Define the fixture that provides the num_steps argument
@pytest.mark.parametrize("num_steps", [2, 9])
@pytest.mark.parametrize("hidden_size", [30, 25])
@pytest.mark.parametrize("init_from_layer", [True, False])
def test_loss_linear(
    num_steps: int, hidden_size: int, init_from_layer: bool
):  # num_steps is now passed by the fixture

    # Define the model
    model = nn.Sequential(
        nn.Linear(30, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 20)  # 0  # 2
    )
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
    ), f"Expected loss to go down for {num_steps} training steps, but it did not. \x1B[3msad sad sad\x1B[23m"