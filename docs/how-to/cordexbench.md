# CORDEXBench workflow

This guide summarizes how to run a reproducible downscaling experiment with the CORDEXBench benchmark dataset.

## Steps

1. **Obtain CORDEXBench data** — follow the data access instructions referenced in the notebook
2. **Open the tutorial notebook** — [`cordexbench_downscaling_deepesd.ipynb`](../tutorials/index.md#cordexbench)
3. **Match preprocessing** — use the same standardization reference period and variable selection as in the notebook
4. **Train DeepESD** — use `DeepESDtas` or `DeepESDpr` depending on the target variable
5. **Evaluate** — report metrics from [`deep4downscaling.metrics`](../user-guide/metrics.md) for comparability

## Tips

- Keep predictor channel order identical between training and inference
- Document the train/validation/test split dates for reproducibility
- Store model checkpoints and configuration (filters, learning rate, epochs) alongside results

## Related

- [DeepESD user guide](../user-guide/models/deepesd.md)
- [Evaluation metrics](../user-guide/metrics.md)
- [Data conventions](../getting-started/data-conventions.md)
