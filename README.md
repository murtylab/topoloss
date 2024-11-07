# topoloss

Induce brain-like topographic structure in your neural networks

![banner](https://github.com/user-attachments/assets/0b8ae5e0-175a-49ee-a690-1b4f89d9d0fd)

```bash
pip install topoloss
```

## Example

```python
import torchvision.models as models
from topoloss.core import TopoLoss, TopoLossConfig, LaplacianPyramid


model = models.resnet18(weights = "DEFAULT")
config = TopoLossConfig(
    layer_wise_configs = [
        LaplacianPyramid(layer_name = 'fc', scale = 1.0, shrink_factor = [3.]),
    ],
)

loss = TopoLoss(
    model = model,
    config = config,
    device = 'cpu'
)

print(loss) ## shows basic info about the objective
print(loss.compute(reduce_mean = True)) ## returns a single number as tensor for backward()
print(loss.compute(reduce_mean = False)) ## returns a dict with layer names as keys
```

## Running tests

```bash
pytest -vvx tests
```
