import math
from .core import TopoLoss

valid_modes = ["linear", "cosine_decay"]

def get_linear_tau(start_value: float, end_value: float, current_step: int, num_steps: int) -> float:
    if current_step >= num_steps:
        return end_value
    return start_value + (end_value - start_value) * (current_step / num_steps)


def get_cosine_decay_tau(start_value: float, end_value: float, current_step: int, num_steps: int) -> float:
    assert end_value < start_value, "cosine_decay mode requires end_value < start_value"
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / num_steps))
    return end_value + (start_value - end_value) * cosine_decay


class TauScheduler:
    def __init__(
        self,
        topo_loss: TopoLoss,
        start_value: float,
        end_value: float,
        num_steps: int,
        mode: str = "linear",
        verbose: bool = False,
    ):
        assert mode in valid_modes, f"mode must be one of {self.valid_modes} but got '{mode}'"
        assert isinstance(topo_loss, TopoLoss), f"topo_loss must be an instance of TopoLoss but got {type(topo_loss)}"

        self.topo_loss = topo_loss
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.mode = mode
        self.verbose = verbose
        self.current_step = 0

    def compute_value(self, current_step: int) -> float:
        if self.mode == "cosine_decay":
            return get_cosine_decay_tau(
            start_value=self.start_value,
            end_value=self.end_value,
            current_step=current_step,
            num_steps=self.num_steps,
            )
        else:  # linear
            return get_linear_tau(
            start_value=self.start_value,
            end_value=self.end_value,
            current_step=current_step,
            num_steps=self.num_steps,
            )

    def step(self, current_step: int = None):
        if current_step is None:
            current_step = self.current_step
        self.current_step += 1

        value = self.compute_value(current_step)
        for loss in self.topo_loss.losses:
            if self.verbose:
                print(f"[TauScheduler/{self.mode}] layer: {loss.layer_name} | {loss.scale:.4f} -> {value:.4f} | step {current_step}/{self.num_steps}")
            loss.scale = value


class ChainedTauScheduler:
    """Chains multiple TauSchedulers sequentially, mirroring PyTorch's ChainedScheduler pattern."""

    def __init__(self, schedulers: list[TauScheduler]):
        assert len(schedulers) > 0, "schedulers list must not be empty"
        self.schedulers = schedulers
        self.current_step = 0

    def step(self):
        # Walk through schedulers in order, consuming steps from each before moving to the next
        remaining = self.current_step
        for scheduler in self.schedulers:
            if remaining < scheduler.num_steps:
                scheduler.step(remaining)
                break
            remaining -= scheduler.num_steps
        else:
            # All schedulers exhausted — keep the last one at its end value
            self.schedulers[-1].step(self.schedulers[-1].num_steps)

        self.current_step += 1
        
    def get_current_tau(self):
        # Return the current tau value from the active scheduler
        remaining = self.current_step
        for scheduler in self.schedulers:
            if remaining < scheduler.num_steps:
                return scheduler.compute_value(remaining)
            remaining -= scheduler.num_steps
        return self.schedulers[-1].compute_value(self.schedulers[-1].num_steps)