# Preprocessing

The [`deep4downscaling.trans`](../api/trans.md) module provides xarray-based utilities shared across model families. These functions implement the standard data operations that every downscaling experiment needs.

## Data cleaning

### `remove_days_with_nans`

Drops time steps where **any** variable has a missing value anywhere in the spatial domain. Useful for daily datasets where incomplete spatial coverage should exclude an entire day.

### `align_datasets`

Aligns two datasets along a shared coordinate (typically `time`) by keeping only overlapping values.

## Normalization

### `standardize` / `undo_standardization`

Normalize data using per-variable mean and standard deviation from a reference dataset (`data_ref`). Always keep the reference dataset so you can undo standardization before writing projections in physical units.

## Format conversion

### `xarray_to_numpy`

Stacks dataset variables into a numpy array suitable for PyTorch. Use `ignore_vars` to exclude auxiliary coordinates.

### `compute_valid_mask` / `compute_valid_multivariate_mask`

Build boolean masks marking valid grid cells. Required for precipitation and other variables where zero or missing values have physical meaning.

## Splitting

### `split_data`

Splits one or more numpy arrays along the time axis into train, validation, and test partitions while keeping indices aligned across arrays.

## Post-processing projections

### `scaling_delta_correction`

Applies a delta-scaling correction to model outputs, useful when combining learned downscaling with bias adjustment strategies.

### `replicate_across_time`

Broadcasts a static dataset across the time dimension of a reference dataset.

### `sort_variables`

Reorders variables in a dataset to match a reference, optionally keeping only shared variables.

## Typical preprocessing chain

```python
import deep4downscaling.trans as trans

x_data = trans.remove_days_with_nans(x_data)
y_data = trans.remove_days_with_nans(y_data)
x_data, y_data = trans.align_datasets(x_data, y_data, "time")

x_data = trans.standardize(x_ref, x_data)
y_data = trans.standardize(y_ref, y_data)

x_train, x_valid, x_test, y_train, y_valid, y_test = trans.split_data(
    trans.xarray_to_numpy(x_data),
    trans.xarray_to_numpy(y_data),
    train_ratio=0.7,
    valid_ratio=0.15,
)
```

Adapt ratios and steps to your experimental design. The [DeepESD tutorial](../tutorials/index.md) shows a complete preprocessing pipeline with domain-specific choices.
