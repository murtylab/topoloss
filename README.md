# topoloss

Induce brain-like topographic structure in your neural networks

![banner](https://github.com/user-attachments/assets/0b8ae5e0-175a-49ee-a690-1b4f89d9d0fd)

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
            layer = model.fc,
            factor_h=3.0, 
            factor_w=3.0, 
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

## Running tests

```bash
pytest -vvx tests
```