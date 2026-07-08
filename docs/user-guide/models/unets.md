# U-Nets

U-Net architectures in `deep4downscaling` provide encoder–decoder convolutional models for gridded downscaling.

## Available models

| Class | Variable | Description |
| --- | --- | --- |
| `UnetTas` | Temperature | U-Net adapted for temperature downscaling |
| `UnetPr` | Precipitation | U-Net adapted for precipitation downscaling |

Both are exported from `deep4downscaling.deep.models`.

## When to use U-Nets

U-Nets are a good choice when:

- You need a **deterministic** gridded regression model
- You prefer an encoder–decoder CNN over the DeepESD head design
- You do not require probabilistic or adversarial training

For probabilistic outputs, consider [DeepESD](deepesd.md) (stochastic) or [NoisyViT](vit.md) (CRPS).

## Training

Use `standard_training_loop` with `MseLoss` or `MaeLoss`. See the [Quickstart](../../getting-started/quickstart.md) for the general training pattern.

## Shared building blocks

U-Net layers reuse components from `deep4downscaling.deep.models.blocks` (`UnitConv`, `UpLayer`), which are also used by other architectures.

## API reference

See [models API](../../api/deep/models.md) for constructor parameters and forward-pass details.
