# Quickstart

This page shows the core steps of a deterministic downscaling workflow. For a complete, runnable example with real data preparation, see the [DeepESD tutorial notebook](../tutorials/index.md#deterministic-deepesd).

## 1. Prepare the data

Load predictor (GCM/reanalysis) and predictand (observations) as `xarray.Dataset` objects, then apply standard preprocessing:

```python
import xarray as xr
import deep4downscaling.trans as trans

# Load your NetCDF files (paths are examples)
x_data = xr.open_dataset("predictors.nc")
y_data = xr.open_dataset("predictand.nc")

# Remove days with missing spatial values
x_data = trans.remove_days_with_nans(x_data)
y_data = trans.remove_days_with_nans(y_data)

# Align time coordinates
x_data, y_data = trans.align_datasets(x_data, y_data, coord="time")

# Standardize predictors and predictand using a reference period
x_data = trans.standardize(x_data_ref, x_data)
y_data = trans.standardize(y_data_ref, y_data)

# Convert to numpy arrays for PyTorch
x_np = trans.xarray_to_numpy(x_data)
y_np = trans.xarray_to_numpy(y_data)
```

See [Data conventions](data-conventions.md) for the expected array shapes and dimension ordering.

## 2. Build a DataLoader

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(
    torch.tensor(x_np, dtype=torch.float32),
    torch.tensor(y_np, dtype=torch.float32),
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

For multivariate or custom sampling logic, use `deep4downscaling.deep.utils.StandardDataset`.

## 3. Instantiate a model

```python
from deep4downscaling.deep.models import DeepESDtas

# x_shape: (time, channels, lat, lon) — channels set at batch dim 1
# y_shape: (time, gridpoints)
model = DeepESDtas(
    x_shape=(None, x_np.shape[1], x_np.shape[2], x_np.shape[3]),
    y_shape=(None, y_np.shape[1]),
    filters_last_conv=25,
    stochastic=False,
)
```

## 4. Train

```python
from deep4downscaling.deep.loss import MseLoss
from deep4downscaling.deep.train import standard_training_loop

loss_fn = MseLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

history = standard_training_loop(
    model=model,
    model_name="deepesd_tas",
    model_path="./models",
    loss_function=loss_fn,
    optimizer=optimizer,
    num_epochs=50,
    device="cuda",
    train_data=train_loader,
    valid_data=valid_loader,
    patience_early_stopping=10,
)
```

## 5. Predict and evaluate

```python
from deep4downscaling.deep.pred import compute_preds_standard
import deep4downscaling.metrics as metrics

preds = compute_preds_standard(
    x_data=x_test,
    model=model,
    device="cuda",
    var_to_pred="tas",
)

score = metrics.rmse(target=y_test, pred=preds, var_target="tas")
```

## Next steps

- [Workflow overview](../user-guide/workflow.md) — full pipeline from preprocessing to projections
- [Model selection](../user-guide/models/overview.md) — choose between DeepESD, U-Net, ViT, or CGAN
- [Evaluation metrics](../user-guide/metrics.md) — which metric to report for your variable
- [Tutorials](../tutorials/index.md) — complete notebook examples
