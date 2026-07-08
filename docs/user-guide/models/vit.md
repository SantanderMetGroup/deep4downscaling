# Vision Transformers

`deep4downscaling` provides Vision Transformer (ViT) architectures for downscaling, including a stochastic variant trained with CRPS-based losses.

## Models

| Class | Description |
| --- | --- |
| `ViT` | Deterministic Vision Transformer |
| `NoisyViT` | Stochastic variant for ensemble / CRPS training |

## Conventions

### Square output grid

The target grid must be **square**: `gridpoints` (the flattened spatial dimension) must be a perfect square (`H_out == W_out`).

### Decoder options

The `decoder` argument selects the upscaling strategy:

| Decoder | Requirement |
| --- | --- |
| `'pixelshuffle'` (default) | Upscaling factor `H_out // H_tokens` must be a **power of 2** |
| `'linear'` | No power-of-2 constraint on upscaling factor |

### Gridpoint ordering

Both decoders flatten output in **row-major `(lat, lon)` order**, matching:

```python
data.stack(gridpoint=("lat", "lon"))
```

When using `CRPSSpectralLoss`, the `H_shape` and `W_shape` arguments must match the stacking order used for the target.

### NoisyViT ensemble mode

`NoisyViT.forward` behaves differently depending on context:

- **Training** or **gradients enabled** — returns a **list** of ensemble members
- **Evaluation with `torch.no_grad()`** — returns a **single** tensor

!!! warning "CRPS validation"
    When validating with the CRPS loss, **do not** wrap inference in `torch.no_grad()`. Otherwise the loss collapses to a single member and probabilistic scores will be incorrect.

## Losses

Pair `NoisyViT` with:

- `CRPSLoss` — standard CRPS objective
- `CRPSSpectralLoss` — CRPS with spectral decomposition (requires correct `H_shape`/`W_shape`)

## Tutorial

See [ViT downscaling](../../tutorials/index.md#vision-transformer-vit) for a precipitation example with CRPS training.

## API reference

See [models API](../../api/deep/models.md) for full parameter documentation.
