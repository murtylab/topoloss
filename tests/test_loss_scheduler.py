from topoloss import TopoLoss, LaplacianPyramid
from topoloss.scheduler import TauScheduler, ChainedTauScheduler
import pytest
import torch
import torch.nn as nn


def make_topo_loss():
    model = nn.Sequential(nn.Linear(30, 20))
    tl = TopoLoss(
        losses=[
            LaplacianPyramid.from_layer(
                model=model, layer=model[0], scale=1.0, factor_h=2.0, factor_w=2.0
            )
        ]
    )
    return model, tl


@pytest.mark.parametrize("num_steps", [10, 50])
def test_linear_reaches_end_value(num_steps: int):
    """Linear tau should equal end_value after num_steps steps."""
    _, tl = make_topo_loss()
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=0.0,
        end_value=1.0,
        num_steps=num_steps,
        mode="linear",
        verbose=False,
    )
    for _ in range(num_steps):
        scheduler.step()

    assert abs(scheduler.get_current_tau() - 1.0) < 1e-5, (
        f"Expected tau=1.0 after {num_steps} steps, got {scheduler.get_current_tau():.6f}"
    )


@pytest.mark.parametrize("num_steps", [10, 50])
def test_cosine_decay_reaches_end_value(num_steps: int):
    """Cosine decay tau should equal end_value after num_steps steps (requires start > end)."""
    _, tl = make_topo_loss()
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=1.0,
        end_value=0.0,
        num_steps=num_steps,
        mode="cosine_decay",
        verbose=False,
    )
    for _ in range(num_steps):
        scheduler.step()

    assert abs(scheduler.get_current_tau() - 0.0) < 1e-5, (
        f"Expected tau=0.0 after {num_steps} steps, got {scheduler.get_current_tau():.6f}"
    )


@pytest.mark.parametrize("mode,start,end", [
    ("linear", 0.3, 1.0),
    ("cosine_decay", 1.0, 0.0),  # cosine_decay requires start > end
])
def test_scheduler_starts_at_start_value(mode: str, start: float, end: float):
    """Tau should equal start_value before any steps."""
    _, tl = make_topo_loss()
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=start,
        end_value=end,
        num_steps=10,
        mode=mode,
        verbose=False,
    )
    assert abs(scheduler.get_current_tau() - start) < 1e-5, (
        f"Expected tau={start} before any steps, got {scheduler.get_current_tau():.6f}"
    )


def test_linear_scheduler_is_monotone():
    """Linear warmup should be strictly increasing."""
    _, tl = make_topo_loss()
    num_steps = 20
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=0.0,
        end_value=1.0,
        num_steps=num_steps,
        mode="linear",
        verbose=False,
    )
    taus = [scheduler.get_current_tau()]
    for _ in range(num_steps):
        scheduler.step()
        taus.append(scheduler.get_current_tau())

    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1], (
            f"Linear scheduler not monotone at step {i}: {taus[i]:.4f} > {taus[i+1]:.4f}"
        )


def test_cosine_decay_is_monotone():
    """Cosine decay should be strictly decreasing."""
    _, tl = make_topo_loss()
    num_steps = 20
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=1.0,
        end_value=0.0,
        num_steps=num_steps,
        mode="cosine_decay",
        verbose=False,
    )
    taus = [scheduler.get_current_tau()]
    for _ in range(num_steps):
        scheduler.step()
        taus.append(scheduler.get_current_tau())

    for i in range(len(taus) - 1):
        assert taus[i] >= taus[i + 1], (
            f"Cosine decay not monotone at step {i}: {taus[i]:.4f} < {taus[i+1]:.4f}"
        )


def test_chained_scheduler_transitions():
    """ChainedTauScheduler should run warmup then decay, hitting expected midpoint."""
    _, tl = make_topo_loss()
    num_steps = 100
    scheduler = ChainedTauScheduler(
        schedulers=[
            TauScheduler(
                topo_loss=tl,
                start_value=0.0,
                end_value=1.0,
                num_steps=num_steps // 2,
                mode="linear",
                verbose=False,
            ),
            TauScheduler(
                topo_loss=tl,
                start_value=1.0,
                end_value=0.0,
                num_steps=num_steps // 2,
                mode="cosine_decay",
                verbose=False,
            ),
        ]
    )

    taus = []
    for _ in range(num_steps):
        taus.append(scheduler.get_current_tau())
        scheduler.step()

    # Should start near 0
    assert taus[0] < 0.1, f"Expected tau near 0 at start, got {taus[0]:.4f}"
    # Should peak near 1 at the midpoint
    assert taus[num_steps // 2 - 1] > 0.9, (
        f"Expected tau near 1 at midpoint, got {taus[num_steps // 2 - 1]:.4f}"
    )
    # Should end near 0
    assert taus[-1] < 0.1, f"Expected tau near 0 at end, got {taus[-1]:.4f}"


def test_scheduler_does_not_exceed_bounds():
    """Linear tau should never go outside [start_value, end_value], even past num_steps."""
    _, tl = make_topo_loss()
    num_steps = 30
    start, end = 0.0, 1.0
    scheduler = TauScheduler(
        topo_loss=tl,
        start_value=start,
        end_value=end,
        num_steps=num_steps,
        mode="linear",
        verbose=False,
    )
    for _ in range(num_steps + 5):  # a few extra steps past the end
        tau = scheduler.get_current_tau()
        assert start - 1e-5 <= tau <= end + 1e-5, (
            f"Tau {tau:.4f} out of bounds [{start}, {end}]"
        )
        scheduler.step()