# Loss functions

Loss functions live in [`deep4downscaling.deep.loss`](../api/deep/loss.md) and are re-exported via the package `__init__`.

## Available losses

| Class | Type | Use with |
| --- | --- | --- |
| `MseLoss` | Deterministic | DeepESD, U-Net, ViT (regression) |
| `MaeLoss` | Deterministic | Any deterministic model |
| `Asym` | Deterministic, asymmetric | Variables where over/under-prediction should be penalized differently |
| `NLLGaussianLoss` | Probabilistic | Stochastic DeepESD / NoisyDeepESD for temperature |
| `NLLBerGammaLoss` | Probabilistic | Stochastic DeepESD / NoisyDeepESD for precipitation |
| `CRPSLoss` | Probabilistic | NoisyViT ensembles |
| `CRPSSpectralLoss` | Probabilistic | NoisyViT with spectral CRPS decomposition |

## Import

```python
from deep4downscaling.deep.loss import MseLoss, NLLGaussianLoss, CRPSLoss
```

## Pairing guide

### Temperature — deterministic

```python
from deep4downscaling.deep.loss import MseLoss
loss_fn = MseLoss()
```

### Temperature — stochastic (Gaussian NLL)

```python
from deep4downscaling.deep.loss import NLLGaussianLoss
loss_fn = NLLGaussianLoss()
```

The model must output mean and log-variance channels (stochastic DeepESD with `predictand="tas"`).

### Precipitation — stochastic (Bernoulli-gamma NLL)

```python
from deep4downscaling.deep.loss import NLLBerGammaLoss
loss_fn = NLLBerGammaLoss()
```

Apply `deep4downscaling.deep.utils.precipitation_NLL_trans` to transform precipitation data before training when required.

### Precipitation — asymmetric penalty

```python
from deep4downscaling.deep.loss import Asym
loss_fn = Asym(threshold=1.0, weight=2.0)
```

### Ensemble — CRPS

```python
from deep4downscaling.deep.loss import CRPSSpectralLoss
loss_fn = CRPSSpectralLoss(H_shape=H, W_shape=W)
```

See [Vision Transformers](models/vit.md) for NoisyViT-specific constraints.

## Adding a custom loss

See the [how-to guide](../how-to/custom-loss.md) for a step-by-step pattern.
