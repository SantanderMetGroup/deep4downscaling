# DeepESD

DeepESD is the flagship architecture in `deep4downscaling`, based on the model proposed by Baño-Medina et al. for CORDEX EUR-44 downscaling.

## Architecture

DeepESD applies three convolutional layers to the predictor field `(channels, lat, lon)`, flattens the result, and maps to the predictand grid via a linear head.

```
Predictor (C, lat, lon) → Conv2d × 3 → Flatten → Linear → Predictand (gridpoints)
```

## Variants

| Class | Predictand | Notes |
| --- | --- | --- |
| `DeepESD` | `tas` or `pr` | Base class with full configuration |
| `DeepESDtas` | Temperature | Convenience wrapper |
| `DeepESDpr` | Precipitation | Applies ReLU on deterministic output by default |
| `NoisyDeepESD` | `tas` or `pr` | Stochastic variant with noise injection |
| `DeepESD_Discriminator` | — | Discriminator for adversarial / CGAN training |

## Deterministic vs. stochastic

Set `stochastic=False` for standard regression (single output per gridpoint).

Set `stochastic=True` for probabilistic heads:

- **Temperature** — Gaussian (mean + log-variance)
- **Precipitation** — Bernoulli-gamma (occurrence probability + shape + scale)

Pair stochastic models with the appropriate NLL loss (`NLLGaussianLoss` or `NLLBerGammaLoss`).

## Shape requirements

```python
model = DeepESD(
    x_shape=(batch, n_channels, n_lat, n_lon),  # 4D
    y_shape=(batch, n_gridpoints),               # 2D
    filters_last_conv=25,
    stochastic=False,
    predictand="tas",
)
```

## Prediction

| Mode | Function |
| --- | --- |
| Deterministic | `compute_preds_standard` |
| Gaussian | `compute_preds_gaussian` |
| Bernoulli-gamma (pr) | `compute_preds_ber_gamma` |

## Reference

Baño-Medina, J., et al. (2022). *Downscaling multi-model climate projection ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44.* Geosci. Model Dev., 15, 6747–6758. [https://doi.org/10.5194/gmd-15-6747-2022](https://doi.org/10.5194/gmd-15-6747-2022)

## Tutorials

- [Deterministic DeepESD](../../tutorials/index.md#deterministic-deepesd)
- [Stochastic DeepESD](../../tutorials/index.md#stochastic-deepesd)
- [CORDEXBench](../../tutorials/index.md#cordexbench)
