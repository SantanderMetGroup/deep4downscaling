# Projections and climate change signals

After training and historical evaluation, downscaling models are applied to **future GCM outputs** to produce high-resolution projections. `deep4downscaling` provides utilities for inference, post-processing, and climate change signal (CCS) analysis.

## Generating projections

### 1. Preprocess future GCM data

Apply the **same preprocessing** used during training:

- Same predictor variables and channel order
- Standardize using the **training-period reference** (`trans.standardize`)
- Same spatial domain and resolution

### 2. Run inference

Use the prediction function matching your model type:

| Model type | Function |
| --- | --- |
| Deterministic | `compute_preds_standard` |
| Gaussian (tas) | `compute_preds_gaussian` |
| Bernoulli-gamma (pr) | `compute_preds_ber_gamma` |

All functions in [`deep4downscaling.deep.pred`](../api/deep/pred.md) accept `xarray.Dataset` inputs and return xarray-compatible outputs.

### 3. Post-process

```python
import deep4downscaling.trans as trans

# Recover physical units
preds = trans.undo_standardization(ref_data, preds)

# Optional: delta-scaling correction
preds = trans.scaling_delta_correction(preds, ...)
```

### 4. Export

Write results to NetCDF for use in climate impact studies or comparison with other downscaling methods.

## Climate change signals (CCS)

The [`deep4downscaling.metrics_ccs`](../api/metrics_ccs.md) module computes differences between future and historical projections after temporal aggregation.

### Reduction functions

Built-in reducers (applied along `time`):

| Function | Quantity |
| --- | --- |
| `mean` | Temporal mean |
| `P02`, `P98` | 2nd and 98th percentiles |
| `TNn` | Mean of annual minimum temperature |
| `TXx` | Mean of annual maximum temperature |
| `R01` | Wet-day frequency |
| `SDII` | Wet-day intensity |
| `RX1day` | Mean of annual maximum 1-day precipitation |

### Computing CCS

```python
import deep4downscaling.metrics_ccs as ccs

signal = ccs.compute_ccs(
    hist_data=hist_projection,
    fut_data=fut_projection,
    reduction_function=ccs.TXx,
    relative=False,
)
```

Set `relative=True` for fractional changes instead of absolute differences.

### Requirements

- Historical and future datasets must share **variable names** and **spatial coordinates**
- Apply the same reduction function to both periods before differencing

## Workflow integration

CCS analysis is stage 6 in the [end-to-end workflow](workflow.md). The DeepESD tutorial notebook covers projection generation from GCM outputs.

## Further reading

- [Data conventions](../getting-started/data-conventions.md)
- [Preprocessing](preprocessing.md)
- [API: metrics_ccs](../api/metrics_ccs.md)
- [API: pred](../api/deep/pred.md)
