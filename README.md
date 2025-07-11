# deep4downscaling

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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

### 2. Create a conda environment from the provided .yml
```bash
conda env create -f requirements/deep4downscaling-gpu.yml
```
If you prefer a CPU-only setup, use the `requirements/deep4downscaling-cpu.yml` file instead.

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

We welcome contributions of all kinds to `deep4downscaling`—from reporting bugs and suggesting improvements to submitting pull requests for new features or fixes. Here’s how you can get involved:

1. **Report Bugs:**  
   If you find an issue, please [open a new GitHub issue](https://github.com/SantanderMetGroup/deep4downscaling/issues) with details on how to reproduce it.

2. **Suggest Enhancements or New Features:**  
   We value user feedback! Feel free to create an issue describing your idea or improvement.

3. **Submit Pull Requests:**  
   - Fork the repository and create a new branch for your changes.  
   - Make your changes or add your new feature.  
   - Open a pull request (PR) to the main branch of `deep4downscaling`.   
