# TopoLoss

Induce brain-like topographic structure in your neural networks. 

![banner](https://github.com/user-attachments/assets/0b8ae5e0-175a-49ee-a690-1b4f89d9d0fd)

Read the [paper](https://arxiv.org/abs/2501.16396) (ICLR 2025), check out the [colab notebook](https://colab.research.google.com/github/toponets/toponets.github.io/blob/main/notebooks/topoloss-demo.ipynb) and play with the [pre-trained models](https://github.com/toponets/toponets) 🤗

```bash
pip install topoloss
```

## Example

```python
import torchvision.models as models
from topoloss import TopoLoss, LaplacianPyramid

model = models.resnet18(weights = "DEFAULT")

topo_loss = TopoLoss(
    losses = [
        LaplacianPyramid.from_layer(
            model=model,
            layer = model.fc, ## supports nn.Linear and nn.Conv2d
            factor_h=8.0, 
            factor_w=8.0, 
            scale = 1.0
        ),
    ],
)
loss = topo_loss.compute(model=model)
## >>> tensor(0.8407, grad_fn=<DivBackward0>)
loss.backward()

loss_dict = topo_loss.compute(model=model, reduce_mean = False) ## {"fc": }
## >>> {'fc': tensor(0.8407, grad_fn=<MulBackward0>)}
```

## Using $\tau$ schedulers

In order to change the strength of the topographic constraint (tau) during training, you can do the following. 

```python
from topoloss.scheduler import TauScheduler, ChainedTauScheduler

## this is a simple linear warmup
scheduler = TauScheduler(
    topo_loss=topo_loss,
    start_value=0.0,
    end_value=1.0,
    num_steps=100,
    mode="linear",
    verbose=False,
)

scheduler.step()
```

You can also chain different schedulers together, much like what you see in pytorch's [`ChainedScheduler`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html)

```python

## linear warmup + cosine decay
scheduler = ChainedTauScheduler(
    schedulers=[
        TauScheduler(
            topo_loss=topo_loss,
            start_value=0.0,
            end_value=1.0,
            num_steps=100,
            mode="linear",
            verbose=False,
        ),
        TauScheduler(
            topo_loss=topo_loss,
            start_value=1.0,
            end_value=0.0,
            num_steps=100,
            mode="cosine_decay",
            verbose=False,
        ),
    ]
)
```

## Running tests

```bash
pytest -vvx tests
```
