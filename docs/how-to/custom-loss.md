# Add a custom loss

This guide shows how to add a new PyTorch loss function that integrates with the existing training loops.

## 1. Create the loss class

Add a new file under `deep4downscaling/deep/loss/` or extend an existing one. Subclass `torch.nn.Module`:

```python
import torch
from torch import nn

class MyCustomLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.mean((pred - target) ** 2)
```

Follow the NumPy docstring style used by existing losses (see `MseLoss` in `standard.py`).

## 2. Export from the package

Add the import and name to `deep4downscaling/deep/loss/__init__.py`:

```python
from .my_custom import MyCustomLoss

__all__ = [..., "MyCustomLoss"]
```

## 3. Use in a training loop

```python
from deep4downscaling.deep.loss import MyCustomLoss
from deep4downscaling.deep.train import standard_training_loop

loss_fn = MyCustomLoss(weight=2.0)

history = standard_training_loop(
    model=model,
    loss_function=loss_fn,
    ...,
)
```

Training loops call `loss_function(pred, target)` (or the signature expected by your loss). Match the output shape of your model to what the loss expects.

## 4. Document

- Add a docstring with `Parameters` and `Returns` sections
- Mention the new loss in `docs/user-guide/losses.md`
- If it enables a new workflow, add a short how-to or notebook example

## Probabilistic losses

If your loss expects multiple output channels (like `NLLGaussianLoss`), ensure the model head produces the correct number of outputs. See [Loss functions](../user-guide/losses.md) for existing patterns.

## Related

- [Loss functions user guide](../user-guide/losses.md)
- [Loss API](../api/deep/loss.md)
- [Contributing](../contributing.md)
