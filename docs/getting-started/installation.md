# Installation

## Requirements

- Python 3.10 or later (3.11 recommended for building the documentation)
- A working PyTorch installation (CPU or CUDA)

The library depends on common scientific Python packages: `numpy`, `pandas`, `xarray`, `netCDF4`, `torch`, `scikit-learn`, `matplotlib`, `cartopy`, and others. See `pyproject.toml` for the full list.

## Install from source

It is recommended to use a virtual environment or a conda environment.

```bash
git clone https://github.com/SantanderMetGroup/deep4downscaling.git
cd deep4downscaling
pip install --upgrade pip setuptools wheel
pip install .
```

### Editable install (development)

```bash
pip install -e .
```

### With documentation tools

To build the documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Verify the installation

```python
import deep4downscaling
print(deep4downscaling.__version__)
```

## Input data

Climate datasets are **not** bundled with the repository due to size constraints. The [tutorials](../tutorials/index.md) describe the expected data format and point to benchmark datasets such as CORDEXBench.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Stable, release-ready code |
| `devel` | Active development and integration |

Contributions should target `devel`. See [Contributing](../contributing.md) for the full workflow.
