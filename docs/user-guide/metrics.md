# Evaluation metrics

The [`deep4downscaling.metrics`](../api/metrics.md) module provides evaluation functions widely used in the statistical downscaling community. All metrics accept `xarray.Dataset` inputs and return xarray objects.

## General guidelines

- **`target`** — observations or reference data (ground truth)
- **`pred`** — model predictions in the same layout as target
- **`var_target`** — variable name to evaluate
- **`season`** — optional filter: `'winter'`, `'summer'`, `'spring'`, `'autumn'`, or `None` for the full year

## Metric catalogue

### Basic errors

| Function | Description | Typical use |
| --- | --- | --- |
| `bias_mean` | Bias of the temporal mean | Overall systematic error |
| `mae` | Mean absolute error | General accuracy |
| `rmse` | Root mean square error | Standard deterministic score |
| `rmse_relative` | RMSE relative to target std | Normalized comparison across domains |
| `corr` | Pearson or Spearman correlation | Linear / monotonic agreement |

### Temperature extremes

| Function | Description |
| --- | --- |
| `bias_tnn` | Bias in annual minimum of daily minimum temperature (TNn) |
| `bias_txx` | Bias in annual maximum of daily maximum temperature (TXx) |
| `diurnal_temp_range` | Diurnal temperature range (DTR) |
| `bias_diurnal_temp_range` | Bias in DTR |

### Precipitation indices

| Function | Description |
| --- | --- |
| `rmse_wet` | RMSE on wet days only |
| `bias_rel_R01` | Relative bias in wet-day frequency (≥ 1 mm) |
| `bias_rel_dry_days` | Relative bias in dry-day proportion |
| `bias_rel_SDII` | Relative bias in wet-day intensity (SDII) |
| `bias_rel_rx1day` | Relative bias in maximum 1-day precipitation (Rx1day) |

### Distributional fidelity

| Function | Description |
| --- | --- |
| `bias_quantile` | Bias at a specified quantile |
| `bias_rel_mean` | Relative bias of the mean |
| `bias_rel_quantile` | Relative bias at a specified quantile |
| `ratio_std` | Ratio of standard deviations |
| `ratio_interannual_var` | Ratio of interannual variability |

### Compound / multivariate

| Function | Description |
| --- | --- |
| `joint_quantile_exceedance` | Joint exceedance probability for two variables |
| `bias_joint_quantile_exceedance` | Bias in joint exceedance |
| `corr_compound` | Correlation between two different variables |
| `bias_corr_compound` | Bias in cross-variable correlation |

### Probabilistic scores

| Function | Description |
| --- | --- |
| `crps_ensemble` | Continuous Ranked Probability Score for ensemble forecasts |
| `normalized_rank` | Normalized rank histogram score |

## Which metrics to report?

| Variable | Recommended minimum set |
| --- | --- |
| Temperature (`tas`) | `bias_mean`, `rmse`, `corr`, `bias_txx`, `bias_tnn` |
| Precipitation (`pr`) | `bias_mean`, `rmse_wet`, `bias_rel_R01`, `bias_rel_SDII`, `bias_rel_rx1day` |
| Probabilistic | `crps_ensemble`, `normalized_rank`, plus deterministic metrics on the ensemble mean |

## Example

```python
import deep4downscaling.metrics as metrics

rmse_score = metrics.rmse(target=obs, pred=pred, var_target="tas")
corr_score = metrics.corr(
    target=obs, pred=pred, var_target="tas",
    method="pearson", deseasonalize=True,
)
```

## Visualization

Use [`deep4downscaling.viz`](../api/viz.md) to produce map plots of metric fields or predicted variables.

## API reference

Full parameter lists are in the [metrics API](../api/metrics.md).
