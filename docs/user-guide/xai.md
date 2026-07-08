# Explainability (XAI)

The [`deep4downscaling.deep.xai`](../api/deep/xai.md) module provides explainability methods adapted for statistical downscaling, helping researchers and decision-makers understand **why** a model produces a given prediction.

## Available methods

| Function | Method | Description |
| --- | --- | --- |
| `compute_ism` | Input Sampling Method (ISM) | Saliency-based attribution by perturbing inputs |
| `compute_asm` | Augmented Sampling Method (ASM) | Extended ISM with additional sampling |
| `compute_sdm` | Source Data Mapping (SDM) | Maps predictand gridpoints to predictor regions |

## Supporting utilities

| Function | Purpose |
| --- | --- |
| `get_grid_position` | Map lat/lon coordinates to flattened gridpoint index |
| `get_station_position` | Map station coordinates to index (station workflows) |
| `postprocess_saliency_torch` | Convert raw saliency tensors to xarray |
| `haversine_distance` | Great-circle distance for spatial mapping |

## Typical workflow

1. Train a downscaling model (e.g. DeepESD)
2. Select a target gridpoint or station and time step
3. Run `compute_ism` or `compute_asm` with the trained model and input data
4. Visualize saliency maps with [`deep4downscaling.viz`](../api/viz.md)

## Tutorial

The [explainability notebook](../tutorials/index.md#explainability) walks through interpreting DeepESD predictions with `deep4downscaling.deep.xai`.

## Dependencies

XAI methods build on [Captum](https://captum.ai/) for gradient-based attribution and use xarray for spatial output.

## API reference

See [XAI API](../api/deep/xai.md) for parameters and return types.
