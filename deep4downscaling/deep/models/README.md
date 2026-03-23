# Deep learning models

This folder contains neural network architectures implemented in `deep4downscaling` for statistical downscaling tasks.

## Organization

- `deepesd/`: DeepESD-based models, including deterministic and stochastic variants, plus discriminator components for adversarial setups.
- `unets/`: U-Net architectures adapted for downscaling.
- `vit/`: Vision Transformer (ViT) architectures, including noisy/stochastic variants.
- `blocks.py`: Shared neural network building blocks used across model families.

The public model classes are exported through `deep4downscaling.deep.models`.
