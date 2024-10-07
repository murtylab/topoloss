# topoloss
topoloss

```
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
loss = topo_loss.compute(model=model)
loss.backward()
print(f"Computed topo loss: {loss.item()}")
```