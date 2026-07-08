# Notebooks

This folder contains example Jupyter notebooks that illustrate typical `deep4downscaling` workflows, from training to evaluation and explainability.

These notebooks are the **tutorial** layer of the documentation. Narrative guides, API reference, and how-tos live in the [`docs/`](../docs/) directory and are published at [deep4downscaling.readthedocs.io](https://deep4downscaling.readthedocs.io).

## Available notebooks

| Notebook | Description |
| --- | --- |
| [`downscaling_deepesd.ipynb`](downscaling_deepesd.ipynb) | End-to-end deterministic DeepESD: training, evaluation, and CCS generation |
| [`downscaling_stochastic_deepesd.ipynb`](downscaling_stochastic_deepesd.ipynb) | Probabilistic DeepESD with NLL loss |
| [`downscaling_cgan.ipynb`](downscaling_cgan.ipynb) | Conditional GAN (Pix2Pix-style) downscaling |
| [`downscaling_deepesd_canary_islands.ipynb`](downscaling_deepesd_canary_islands.ipynb) | DeepESD adapted to the Canary Islands domain |
| [`explainability_deepesd.ipynb`](explainability_deepesd.ipynb) | XAI workflow with `deep4downscaling.deep.xai` |
| [`cordexbench_downscaling_deepesd.ipynb`](cordexbench_downscaling_deepesd.ipynb) | CORDEXBench benchmark example |
| [`downscaling_vit.ipynb`](downscaling_vit.ipynb) | Stochastic ViT downscaling with CRPS loss |

## Suggested learning path

1. `downscaling_deepesd.ipynb` — learn the full pipeline
2. [Data conventions](../docs/getting-started/data-conventions.md) — read alongside the notebook
3. Branch to stochastic, ViT, or CGAN notebooks as needed

## Notes

- Input climate datasets are not stored in this repository due to size constraints.
- These notebooks are practical templates and can be adapted to new domains and variables.
