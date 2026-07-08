# Conditional GANs

`deep4downscaling` supports adversarial downscaling setups using a generator–discriminator architecture in the Pix2Pix style.

## Components

| Component | Class |
| --- | --- |
| Generator | `DeepESD` (or compatible CNN generator) |
| Discriminator | `DeepESD_Discriminator` |

## Training

Use `standard_cgan_training_loop` from [`deep4downscaling.deep.train`](../../api/deep/train.md), which alternates generator and discriminator updates.

For DeepESD with adversarial loss but without the full CGAN setup, `adversarial_training_loop` is also available.

## When to use

CGAN training may improve spatial realism of downscaled fields by penalizing unrealistic structures through the discriminator. It adds training complexity compared to standard supervised learning.

Evaluate carefully with the [metrics](../metrics.md) relevant to your variable — improved visual realism does not always translate to better distributional scores.

## Tutorial

See [CGAN downscaling](../../tutorials/index.md#conditional-gan-cgan) for an end-to-end notebook example.

## API reference

- [Training loops](../../api/deep/train.md)
- [Models](../../api/deep/models.md)
