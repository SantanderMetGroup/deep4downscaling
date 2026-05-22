<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg" width="450">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo-light.svg" width="450">
    <img alt="deep4downscaling logo" src="docs/logo-light.svg" width="450">
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19355157.svg)](https://doi.org/10.5281/zenodo.19355157)

## Description

`deep4downscaling` is a Python library designed for developing deep learning models for statistical downscaling. The library focuses on two main objectives:

**Ease of Research**:
`deep4downscaling` provides a suite of standard techniques essential for the statistical downscaling field, such as:

- Data Preprocessing 
- Data transformations
- Standardization and Normalization

By offering these foundational methods, researchers can focus on innovative tasks such as novel architecture development, rather than re-implementing standard routines from scratch.

**Established Deep Learning Models**:
`deep4downscaling` also supplies established deep learning models that can be used to downscale global climate model outputs. Beyond the models themselves, the library provides:

- Tools to compute and generate projections (ensuring compatibility with standard climate data formats like NetCDF).
- Scripts for proper post-processing (e.g., bias correction, domain mapping).

In addition to these main goals, `deep4downscaling` includes:

**Comprehensive Evaluation Metrics**:
A dedicated collection of evaluation metrics widely recognized in the downscaling community, enabling researchers to thoroughly assess model performance.

| Metric | Description |
| --- | --- |
| `bias_mean` | Bias of the mean between the target and predicted datasets. |
| `bias_tnn` | Bias of the annual minimum of daily minimum temperature (TNn). |
| `bias_txx` | Bias of the annual maximum of daily maximum temperature (TXx). |
| `bias_quantile` | Bias of a specified quantile. |
| `mae` | Mean Absolute Error (MAE). |
| `rmse` | Root Mean Square Error (RMSE). |
| `rmse_wet` | RMSE for wet days only. |
| `rmse_relative` | RMSE relative to the target's standard deviation. |
| `bias_rel_mean` | Relative bias of the mean. |
| `bias_rel_quantile` | Relative bias of a specified quantile. |
| `bias_rel_R01` | Relative bias of the wet-day frequency index (R01). |
| `bias_rel_dry_days` | Relative bias of the proportion of dry days. |
| `bias_rel_SDII` | Relative bias of the wet-day intensity index (SDII). |
| `bias_rel_rx1day` | Relative bias of the maximum 1-day precipitation index (Rx1day). |
| `ratio_std` | Ratio of standard deviations. |
| `ratio_interannual_var` | Ratio of interannual variability. |
| `corr` | Pearson or Spearman correlation, with an option for deseasonalized data. |
| `joint_quantile_exceedance` | Joint exceedance probability for a given quantile for two variables. |
| `bias_joint_quantile_exceedance` | Bias in the joint quantile exceedance probability. |
| `diurnal_temp_range` | Diurnal temperature range (DTR). |
| `bias_diurnal_temp_range` | Bias in the diurnal temperature range. |
| `corr_compound` | Pearson or Spearman correlation between two different variables. |
| `bias_corr_compound` | Bias in the correlation between two different variables. |
| `crps_ensemble` | Continuous Ranked Probability Score (CRPS) for an ensemble forecast. |
| `normalized_rank` | Normalized Rank for an ensemble forecast. |

**eXplainable Artificial Intelligence (XAI) Techniques Tailored for Downscaling**:
XAI techniques adapted for statistical downscaling models. This ensures that generated projections are transparent and trustworthy—a critical feature for decision-makers and other end-users who rely on climate modeling outputs.

By combining core data transformations, established deep learning models, advanced evaluation metrics, and explainable AI, `deep4downscaling` aims to empower the research community to develop and validate cutting-edge downscaling solutions with greater efficiency and confidence.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SantanderMetGroup/deep4downscaling/
cd deep4downscaling
```

### 2. Install the library
It is recommended to use a virtual environment or a conda environment. Once your environment is active, you can install the library and its dependencies using `pip`:

```bash
pip install .
```

For development purposes, you can install the library in editable mode:

```bash
pip install -e .
```

## Usage

We provide a set of Jupyter notebooks in the `notebooks` directory that demonstrate the basic functionality of the `deep4downscaling` library. These notebooks cover topics such as:

- Data preprocessing and transformations  
- Model training and evaluation  
- Using the built-in metrics and explainable AI features  

As new features are developed and added to `deep4downscaling`, additional example notebooks will be included to help you stay up-to-date with the latest capabilities.

## Documentation

While `deep4downscaling` does not currently offer a formal documentation website, all library functions include comprehensive `docstrings` describing their purpose, parameters, and return values. This ensures that the code is self-explanatory for developers who want to use or extend the library.

For further guidance on how to use `deep4downscaling`, please refer to:
- The notebooks in `notebooks`, which provide example workflows.  
- The `docstrings` in the source code, which offer detailed explanations of functions and classes.  

Should you have any questions or need clarifications, feel free to open an issue or contribute to improving the documentation.

## Contributing

We use two main branches:

- `main`: stable, release-ready code (default branch when cloning).
- `devel`: active development / integration branch.

All pull requests must target `devel`. Direct pushes to `main` and `devel` are restricted to the maintainers. Maintainers periodically merge `devel` into `main` and create new tagged releases from `main`.

---

### For collaborators (with write access)

1. Clone the repository:

   ```bash
   git clone https://github.com/SantanderMetGroup/deep4downscaling.git
   cd deep4downscaling
   ```

2. Create a `feature`/`fix` branch from `devel`:

   ```bash
   git checkout devel
   git pull origin devel
   git checkout -b feature/my-change
   ```

3. Work and keep in sync with `devel` (optional but recommended):

   ```bash
   git checkout devel
   git pull origin devel
   git checkout feature/my-change
   git merge devel
   ```

4. Push and open a pull request into `devel`:

   ```bash
   git push -u origin feature/my-change
   ```

On GitHub, open a PR with base branch `devel` (not `main`).

### For external contributors (no write access)

1. Fork this repository on GitHub to your own account.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/deep4downscaling.git
   cd deep4downscaling
   ```

3. Add the original repo as upstream and fetch:

   ```bash
   git remote add upstream https://github.com/SantanderMetGroup/deep4downscaling.git
   git fetch upstream
   ```

4. Create a `feature`/`fix` branch from `upstream/devel`:

   ```bash
   git checkout -b feature/my-change upstream/devel
   ```

5. Commit your changes and push to your fork:

   ```bash
   git add ...
   git commit -m "Describe your change"
   git push -u origin feature/my-change
   ```

6. Open a pull request to this repository:

   - base repository: `SantanderMetGroup/deep4downscaling`
   - base branch: `devel`
   - head repository: `<your-username>/deep4downscaling`
   - compare branch: `feature/my-change`
