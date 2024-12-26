# topoloss

Induce brain-like topographic structure in your neural networks

![banner](https://github.com/user-attachments/assets/0b8ae5e0-175a-49ee-a690-1b4f89d9d0fd)

```bash
pip install topoloss
```

## Example

```python
import torchvision.models as models
from topoloss import TopoLoss, LaplacianPyramidLoss


model = models.resnet18(weights = "DEFAULT")

loss = TopoLoss(
    losses = [
        LaplacianPyramidLoss(layer_name = 'fc',factor_h=3.0, factor_w=3.0),
    ],
)

print(loss) ## shows basic info about the objective
print(loss.compute(model=model, reduce_mean = True)) ## returns a single number as tensor for backward()
print(loss.compute(model=model, reduce_mean = False)) ## returns a dict with layer names as keys
```

## Running tests

```bash
pytest -vvx tests
```