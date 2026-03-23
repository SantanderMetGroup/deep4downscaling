# Notebooks

This folder contains example Jupyter notebooks that illustrate typical `deep4downscaling` workflows, from training to evaluation and explainability.

## Available notebooks

- `downscaling_deepesd.ipynb`: End-to-end deterministic downscaling with DeepESD, including training, test-period evaluation, and climate change signal generation from GCM outputs.
- `downscaling_stochastic_deepesd.ipynb`: Probabilistic (stochastic) DeepESD workflow using a probabilistic loss.
- `downscaling_cgan.ipynb`: Downscaling with a conditional GAN (Pix2Pix-style setup).
- `downscaling_deepesd_canary_islands.ipynb`: DeepESD example adapted to the Canary Islands domain.
- `explainability_deepesd.ipynb`: XAI workflow for interpreting DeepESD predictions with `deep4downscaling.deep.xai`.
- `cordexbench_downscaling_deepesd.ipynb`: Example using the CORDEXBench benchmark dataset.
- `downscaling_vit.ipynb`: Downscaling precipitation with a stochastic ViT (CRPS)

## Notes

- Input climate datasets are not stored in this repository due to size constraints.
- These notebooks are meant as practical templates and can be adapted to new domains and variables.
