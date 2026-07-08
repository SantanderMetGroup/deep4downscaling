# deep4downscaling

`deep4downscaling` is a Python library for developing deep learning models for **statistical downscaling** of global climate model (GCM) outputs.

## What it provides

- **Data preprocessing** — xarray-based transforms, standardization, and alignment utilities
- **Deep learning models** — DeepESD, U-Nets, Vision Transformers, and CGAN setups
- **Training & inference** — training loops, prediction helpers, and NetCDF-compatible outputs
- **Evaluation metrics** — metrics widely used in the downscaling community
- **Climate change signals** — tools to assess projected changes between historical and future periods
- **Explainability (XAI)** — interpretation methods tailored to downscaling models

## Documentation map

This site follows the [Diátaxis](https://diataxis.fr/) framework:

| Section | Purpose |
| --- | --- |
| [Getting started](getting-started/installation.md) | Install the library and run a minimal example |
| [User guide](user-guide/workflow.md) | Understand concepts, conventions, and design choices |
| [Tutorials](tutorials/index.md) | End-to-end Jupyter notebook workflows |
| [How-to guides](how-to/index.md) | Solve specific tasks quickly |
| [API reference](api/index.md) | Lookup functions, classes, and parameters |

## Quick install

```bash
git clone https://github.com/SantanderMetGroup/deep4downscaling.git
cd deep4downscaling
pip install --upgrade pip setuptools wheel
pip install .
```

For development:

```bash
pip install -e ".[docs]"
```

## Citation

If you use `deep4downscaling` in your research, please cite the library via its [Zenodo DOI](https://doi.org/10.5281/zenodo.19355157).

## License

MIT — see the [LICENSE](https://github.com/SantanderMetGroup/deep4downscaling/blob/docs/LICENSE) file in the repository.
