import sys
import torch
import numpy as np
import xarray as xr

import deep4downscaling.trans as trans

def _predict(model: torch.nn.Module, device: str, **data: np.ndarray) -> np.ndarray:

    """
    Internal function to compute the prediction of a certain DL model given
    some input data. This function is able to handle DL models with any number
    of inputs to their forward() method.

    Parameters
    ----------
    model : torch.nn.Module
        Model used to compute the predictions.
    
    device : str
        Device used to run the inference (cuda or cpu).

    data : np.ndarray
        Input/Inputs to the model. There are no restrictions for the 
        argument name.

    Returns
    -------
    np.ndarray
    """

    model = model.to(device)

    for key, value in data.items():
        data[key] = torch.tensor(data[key]).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(*data.values())

    y_pred = y_pred.cpu().numpy()

    return y_pred

def _pred_to_xarray(data_pred: np.ndarray, time_pred: np.ndarray,
                    var_target: str, mask: xr.Dataset) -> xr.Dataset:

    """
    This internal function transforms the prediction from a DL model
    (np.ndarray) into a xr.Dataset with the corresponding temporal and
    spatial dimensions. To do so it takes as input a mask (xr.Dataset)
    defining where to introduce (spatial dimension) the predictions of
    the DL model, returning as output a xr.Dataset filled with the
    data_pred. This function requires the time dimension of the 
    output xr.Dataset, which should be easily obtained as the prediction
    of the DL model needs to be computed from anotehr xr.Dataset.

    Parameters
    ----------
    data_pred : np.ndarray
        Predictions of a DL model. Generally these are computed using the
        _predict internal function.

    time_pred : np.ndarray
        Array containing the temporal coordinated of the output xr.Dataset
        (in datetime64[ns]).

    var_target: str
        Target variable.

    mask: xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    Returns
    -------
    xr.Dataset
        The data_pred argument properly transformed to xr.Dataset.
    """

    # Expand the mask in the time dimension of the final prediction
    mask = mask.expand_dims(time=time_pred)
    mask = mask.ffill(dim='time')

    # By default xarray casts the mask to int64 but we need a 
    # float-based type
    mask[var_target] = mask[var_target].astype('float32')
    
    # Stack following the procedure perform in all these modules
    # ('lat', 'lon')
    mask = mask.stack(gridpoint=('lat', 'lon'))

    # Assign the perdiction to the gridpoints of the mask with value 1
    # For the 0 ones we assign them np.nan
    one_indices = (mask[var_target].values == 1)
    mask[var_target].values[one_indices] = data_pred.flatten()
    mask[var_target].values[~one_indices] = np.nan

    # Unstack and return
    mask = mask.unstack()
    return mask

def compute_preds_standard(x_data: xr.Dataset, model: torch.nn.Module, device: str,
                           var_target: str, mask: xr.Dataset) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function is
    designed to work with models computing the final prediction
    (e.g., MSE-based models).

    Notes
    -----
    For this function the mask is key, as it allows to convert the raw output of
    the model to the proper xr.Dataset representation.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    Returns
    -------
    xr.Dataset
        The final prediction
    """
    
    x_data_arr = trans.xarray_to_numpy(x_data)
    time_pred = x_data['time'].values

    data_pred = _predict(model=model, device=device, x_data=x_data_arr)
    data_pred = _pred_to_xarray(data_pred=data_pred, time_pred=time_pred,
                                var_target=var_target, mask=mask)

    return data_pred

def compute_preds_gaussian(x_data: xr.Dataset, model: torch.nn.Module, device: str,
                           var_target: str, mask: xr.Dataset) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function
    tailors the prediction of DL models trained to minimize the NLL of a Gaussian
    distribution.

    Notes
    -----
    For this function the mask is key, as it allows to convert the raw output of
    the model to the proper xr.Dataset representation.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    Returns
    -------
    xr.Dataset
        The final prediction
    """
    
    x_data_arr = trans.xarray_to_numpy(x_data)
    time_pred = x_data['time'].values

    data_pred = _predict(model=model, device=device, x_data=x_data_arr)

    # Get the parameters of the Gaussian dist.
    dim_target = data_pred.shape[1] // 2
    mean = data_pred[:, :dim_target]
    log_var = data_pred[:, dim_target:]

    # Compute the prediction
    s_dev = np.exp(log_var) ** (1/2)
    data_pred_final = np.random.normal(loc=mean, scale=s_dev)

    data_pred = _pred_to_xarray(data_pred=data_pred_final, time_pred=time_pred,
                                var_target=var_target, mask=mask)

    return data_pred

def compute_preds_ber_gamma(x_data: xr.Dataset, model: torch.nn.Module, threshold: float,
                            device: str, var_target: str, mask: xr.Dataset) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function
    tailors the prediction of DL models trained to minimize the NLL of a Bernoulli
    and gamma distributions.

    Notes
    -----
    For this function the mask is key, as it allows to convert the raw output of
    the model to the proper xr.Dataset representation.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    threshold : float
        The value used as threshold to define the precipitation for fitting the
        gamma distribution (deep4downscaling.utils.precipitation_NLL_trans). This
        is required to correct the effect of this transformation in the final
        prediction.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    Returns
    -------
    xr.Dataset
        The final prediction
    """

    x_data_arr = trans.xarray_to_numpy(x_data)
    time_pred = x_data['time'].values

    data_pred = _predict(model=model, device=device, x_data=x_data_arr)

    # Get the parameters of the Bernoulli and gamma dists.
    dim_target = data_pred.shape[1] // 3
    p = data_pred[:, :dim_target]
    shape = np.exp(data_pred[:, dim_target:(dim_target*2)])
    scale = np.exp(data_pred[:, (dim_target*2):])

    # Compute the ocurrence
    p_random = np.random.uniform(0, 1, p.shape)
    ocurrence = (p >= p_random) * 1 

    # Compute the amount
    amount = np.random.gamma(shape=shape, scale=scale)

    # Correct the amount
    epsilon = 1e-06
    threshold = threshold - epsilon
    amount = amount + threshold

    # Combine ocurrence and amount
    data_pred_final = ocurrence * amount

    data_pred = _pred_to_xarray(data_pred=data_pred_final, time_pred=time_pred,
                                var_target=var_target, mask=mask)

    return data_pred