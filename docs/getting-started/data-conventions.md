# Data conventions

Consistent data layout is essential for downscaling workflows. This page documents the conventions expected by `deep4downscaling` preprocessing, models, and prediction utilities.

## xarray as the primary interface

Most high-level functions accept `xarray.Dataset` objects. NetCDF is the typical on-disk format. The library operates on daily (or sub-daily) gridded climate data with at least `time`, `lat`, and `lon` coordinates.

## Predictor (GCM / reanalysis) layout

Convolutional models such as DeepESD and U-Nets expect predictors with **4 dimensions**:

```
(time, channels, lat, lon)
```

When converting to PyTorch tensors via `trans.xarray_to_numpy`, variables are stacked along the channel dimension. The exact channel order must remain consistent between training and inference.

## Predictand (observations) layout

Gridded predictands are stored as **2D arrays** after flattening the spatial grid:

```
(time, gridpoints)
```

Create this layout with xarray stacking:

```python
y_stacked = y_data.stack(gridpoint=("lat", "lon"))
```

The default flattening order is **row-major `(lat, lon)`**, which matches the order used by ViT decoders and `CRPSSpectralLoss`.

!!! warning "ViT square grid requirement"
    Vision Transformer models assume a **square** output grid: `gridpoints` must be a perfect square (`H_out == W_out`). See [Vision Transformers](../user-guide/models/vit.md) for decoder-specific constraints.

## Standardization

Use `trans.standardize(data_ref, data)` to normalize using statistics computed on a reference period (typically the training climatology). Undo with `trans.undo_standardization` before writing physical-unit outputs.

Predictors and predictand should be standardized **separately**, each with its own reference dataset.

## Valid masks

For precipitation and other variables with structural zeros or missing grid cells, compute a valid mask before training:

```python
mask = trans.compute_valid_mask(y_data)
# or, for multiple datasets:
mask = trans.compute_valid_multivariate_mask(y_data, x_data)
```

## Train / validation / test splits

`trans.split_data` splits numpy arrays along the time dimension. It supports multiple arrays (predictor, predictand, etc.) and keeps indices aligned.

## Prediction outputs

Functions in `deep4downscaling.deep.pred` convert model outputs back to `xarray.Dataset` objects compatible with NetCDF export. When stacking gridpoints, use the same `(lat, lon)` order as during training.

## Multivariate setups

Some workflows combine multiple predictands (e.g. `tasmin` and `tasmax`). Ensure variable names and ordering are consistent across datasets using `trans.sort_variables`.

## Station-based predictands

Station workflows use a `gridpoint` (or station index) dimension instead of a full lat/lon grid. The XAI module provides `get_station_position` for this layout. See the explainability tutorial for an example.

## Climate change signal (CCS) data

For CCS analysis (`metrics_ccs.compute_ccs`), historical and future projections must share the same variable names and spatial coordinates. Reduction functions (e.g. `mean`, `TXx`, `R01`) are applied along `time` before differencing.

## Checklist before training

1. Predictor arrays are `(time, channels, lat, lon)`
2. Predictand arrays are `(time, gridpoints)` with consistent `(lat, lon)` stacking
3. Time coordinates are aligned between predictor and predictand
4. Standardization references match the training period
5. Missing days are removed or masked consistently
6. For ViT: output grid is square and upscaling factor constraints are met
