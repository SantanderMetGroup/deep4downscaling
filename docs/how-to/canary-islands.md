# Canary Islands domain

This guide covers adapting a standard DeepESD workflow to a regional domain, using the Canary Islands example notebook as a template.

## Steps

1. **Prepare regional predictors and predictands** — subset GCM and observation data to the Canary Islands domain
2. **Follow the notebook** — [`downscaling_deepesd_canary_islands.ipynb`](../tutorials/index.md#canary-islands-domain)
3. **Check grid dimensions** — ensure predictand stacking produces the expected number of gridpoints
4. **Tune `filters_last_conv`** — smaller domains may benefit from fewer convolutional filters

## Regional considerations

- **Coastal effects** — verify land/sea mask handling in your input data
- **Domain size** — very small domains have fewer gridpoints; monitor for overfitting
- **Variable choice** — precipitation and temperature may need different model classes (`DeepESDpr` vs `DeepESDtas`)

## Related

- [Preprocessing](../user-guide/preprocessing.md)
- [DeepESD](../user-guide/models/deepesd.md)
- [Data conventions](../getting-started/data-conventions.md)
