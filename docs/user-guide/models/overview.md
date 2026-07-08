# Model overview

`deep4downscaling` provides several neural network families for statistical downscaling. All models consume gridded predictors and produce gridded (or station) predictands.

## Comparison

| Model | Architecture | Deterministic | Probabilistic | Best for |
| --- | --- | --- | --- | --- |
| [DeepESD](deepesd.md) | CNN + linear head | ✓ | ✓ (NLL) | Established baseline; tas and pr; CORDEX workflows |
| [U-Nets](unets.md) | Encoder–decoder CNN | ✓ | — | Gridded regression; tas (`UnetTas`) and pr (`UnetPr`) |
| [ViT](vit.md) | Vision Transformer | ✓ | ✓ (NoisyViT + CRPS) | Square output grids; spectral CRPS loss |
| [CGAN](cgan.md) | Generator + discriminator | ✓ | — | Adversarial training with `DeepESD_Discriminator` |

## Public API

Model classes are exported from `deep4downscaling.deep.models`:

```python
from deep4downscaling.deep.models import (
    DeepESD, DeepESDtas, DeepESDpr,
    NoisyDeepESD,
    DeepESD_Discriminator,
    UnetTas, UnetPr,
    ViT, NoisyViT,
)
```

## Choosing a model

**Start with DeepESD** if you want a well-documented baseline aligned with published CORDEX downscaling work. Use `DeepESDtas` or `DeepESDpr` for convenience, or the base `DeepESD` class for full control.

**Use U-Nets** for classic encoder–decoder spatial refinement when you do not need probabilistic outputs.

**Use ViT / NoisyViT** when your output domain is a square grid and you want transformer-based architectures or CRPS-trained ensembles.

**Use CGAN** when adversarial training with a discriminator may improve spatial realism (see the CGAN tutorial).

## Pairing models with losses

| Setup | Model | Loss |
| --- | --- | --- |
| Deterministic regression | DeepESD, U-Net, ViT | `MseLoss`, `MaeLoss`, `Asym` |
| Probabilistic temperature | DeepESD (stochastic), NoisyDeepESD | `NLLGaussianLoss` |
| Probabilistic precipitation | DeepESD (stochastic), NoisyDeepESD | `NLLBerGammaLoss` |
| Ensemble / CRPS | NoisyViT | `CRPSLoss`, `CRPSSpectralLoss` |

See [Loss functions](../losses.md) for details.

## Training loops

| Model type | Training function |
| --- | --- |
| Standard supervised | `standard_training_loop` |
| DeepESD + discriminator | `adversarial_training_loop` |
| CGAN (Pix2Pix) | `standard_cgan_training_loop` |

## Further reading

- [DeepESD](deepesd.md)
- [U-Nets](unets.md)
- [Vision Transformers](vit.md)
- [Conditional GANs](cgan.md)
- [API reference: models](../../api/deep/models.md)
