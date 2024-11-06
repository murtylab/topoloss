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

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=False)

# define where to apply the topo loss
topo_loss = TopoLoss(
    model=model,
    losses=[
        LaplacianPyramidLoss.from_layer(
            model=model,
            layer=model.layer3[1].conv2,
            factor_h=3.0,
            factor_w=3.0,
        )
        ## add more layers here if you want :)
    ]
)

# Compute the loss
loss = topo_loss.compute(reduce_mean=True)
loss.backward()
print(f"Computed topo loss: {loss.item()}")
```

## Running tests

```bash
pytest -vvx tests
```
