# Tutorials

End-to-end Jupyter notebooks live in the [`notebooks/`](https://github.com/SantanderMetGroup/deep4downscaling/tree/docs/notebooks) directory of the repository. They are the primary **tutorial** layer of the documentation.

!!! note "Input data"
    Climate datasets are not stored in the repository due to size constraints. Each notebook documents how to obtain or prepare the required input files.

## Deterministic DeepESD

**Notebook:** [`downscaling_deepesd.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/downscaling_deepesd.ipynb)

End-to-end deterministic downscaling with DeepESD: training, test-period evaluation, and climate change signal generation from GCM outputs.

**Covers:** preprocessing → training → evaluation → future projections

## Stochastic DeepESD

**Notebook:** [`downscaling_stochastic_deepesd.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/downscaling_stochastic_deepesd.ipynb)

Probabilistic DeepESD workflow using negative log-likelihood losses for temperature or precipitation.

**Covers:** stochastic model setup → NLL training → probabilistic evaluation

## Conditional GAN (CGAN)

**Notebook:** [`downscaling_cgan.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/downscaling_cgan.ipynb)

Downscaling with a conditional GAN (Pix2Pix-style generator–discriminator setup).

**Covers:** adversarial training → spatial evaluation

## Vision Transformer (ViT)

**Notebook:** [`downscaling_vit.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/downscaling_vit.ipynb)

Downscaling precipitation with a stochastic ViT trained using CRPS loss.

**Covers:** ViT conventions → CRPS training → ensemble evaluation

## Explainability

**Notebook:** [`explainability_deepesd.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/explainability_deepesd.ipynb)

XAI workflow for interpreting DeepESD predictions.

**Covers:** ISM / ASM saliency → spatial attribution maps

## CORDEXBench

**Notebook:** [`cordexbench_downscaling_deepesd.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/cordexbench_downscaling_deepesd.ipynb)

Reproducible example using the CORDEXBench benchmark dataset.

**Covers:** benchmark data setup → DeepESD training → standardized evaluation

## Canary Islands domain

**Notebook:** [`downscaling_deepesd_canary_islands.ipynb`](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/notebooks/downscaling_deepesd_canary_islands.ipynb)

DeepESD adapted to the Canary Islands domain — an example of regional customization.

**Covers:** domain-specific preprocessing → regional training

## Suggested learning path

1. Start with **Deterministic DeepESD** to learn the full pipeline
2. Read [Data conventions](../getting-started/data-conventions.md) alongside the notebook
3. Branch to **Stochastic DeepESD** or **ViT** if you need probabilistic outputs
4. Explore **Explainability** once you have a trained model
5. Use **CORDEXBench** for benchmark comparisons

## Running notebooks locally

```bash
git clone https://github.com/SantanderMetGroup/deep4downscaling.git
cd deep4downscaling
pip install -e .
jupyter lab notebooks/
```
