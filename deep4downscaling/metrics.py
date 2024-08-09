import xarray as xr

def _filter_by_season(data : xr.Dataset, season : str) -> xr.Dataset:

    """
    Internal function to filter a xr.Dataset with respect to the
    provided season.

    Parameters
    ----------
    data : xr.Dataset
        Data to filter

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    if season is None:
        pass
    elif season == 'winter':
        data = data.where(data['time.season'] == 'DJF', drop=True)
    elif season == 'summer':
        data = data.where(data['time.season'] == 'JJA', drop=True)
    elif season == 'spring':
        data = data.where(data['time.season'] == 'MAM', drop=True)
    elif season == 'autumn':
        data = data.where(data['time.season'] == 'SON', drop=True)

    return data

def bias_mean(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the mean (across time) between the target and pred
    datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.mean('time') - target.mean('time'))
    return metric

def bias_tnn(target: xr.Dataset, pred: xr.Dataset,
             var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the annual minimum of daily minimum temperature (TNn)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target = target.groupby('time.year').min('time')
    pred = pred.groupby('time.year').min('time')

    metric = (pred.mean('year') - target.mean('year'))
    return metric

def bias_txx(target: xr.Dataset, pred: xr.Dataset,
             var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the annual maximum of daily maximum temperature (TXx)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target = target.groupby('time.year').max('time')
    pred = pred.groupby('time.year').max('time')

    metric = (pred.mean('year') - target.mean('year'))
    return metric

def bias_quantile(target: xr.Dataset, pred: xr.Dataset, quantile: float,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the specified quantile (across time) between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    quantile : float
        Quantile on which the bias is computed [0,1]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.quantile(quantile, 'time') - target.quantile(quantile, 'time'))
    return metric

def mae(target: xr.Dataset, pred: xr.Dataset,
        var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the mean absolute error between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = abs(pred - target).mean('time')
    return metric

def rmse(target: xr.Dataset, pred: xr.Dataset,
         var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = ((pred - target) ** 2).mean('time') ** (1/2)
    return metric

def rmse_wet(target: xr.Dataset, pred: xr.Dataset,
             var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between the target and
    pred datasets for the wet days (>=1 mm).

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target = target.where(target[var_target] >= 1)
    pred = pred.where(pred[var_target] >= 1)

    metric = ((pred - target) ** 2).mean('time') ** (1/2)
    return metric

def rmse_relative(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between
    the target and pred datasets relative to the target's
    standard deviation.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_std = target.std('time')

    metric = (((pred - target) ** 2).mean('time') ** (1/2)) / target_std
    return metric

def bias_rel_mean(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the mean (across time) between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_mean = target.mean('time')

    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_quantile(target: xr.Dataset, pred: xr.Dataset, quantile: float,
                      var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the specified quantile (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    quantile : float
        Quantile on which the bias is computed [0,1]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_quantile = target.quantile(quantile, 'time')

    metric = (pred.quantile(quantile, 'time') - target_quantile) / target_quantile
    metric = metric * 100
    return metric

def bias_rel_R01(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                 threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the R01 index (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Compute proportion of wet days
    target = (target >= threshold) * 1
    pred = (pred >= threshold) * 1

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_dry_days(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                      threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the proportion of dry days (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Compute proportion of wet days
    target = (target < threshold) * 1
    pred = (pred < threshold) * 1

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_SDII(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                  threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the SDII index (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Filter wet days
    target = target.where(target[var_target] >= threshold)
    pred = pred.where(pred[var_target] >= threshold)

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_rx1day(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the Rx1day index between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target = target.groupby('time.year').max('time')
    pred = pred.groupby('time.year').max('time')

    target_mean = target.mean('year')

    metric = (pred.mean('year') - target_mean) / target_mean
    metric = metric * 100
    return metric

def ratio_std(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the ratio of standard deviations
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_std = target.std('time')
    pred_std = pred.std('time')

    metric = pred_std / target_std
    return metric

def ratio_interannual_var(target: xr.Dataset, pred: xr.Dataset,
                          var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the ratio of the interannual variatiability
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_inter_var = target[var_target].groupby('time.year').mean(dim='time').std('year')
    pred_inter_var = pred[var_target].groupby('time.year').mean(dim='time').std('year')

    metric = pred_inter_var / target_inter_var
    metric = metric.to_dataset()
    return metric

def corr_pearson(target: xr.Dataset, pred: xr.Dataset, deseasonal: bool,
                 var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the correlation (pearson) between the target and pred
    datasets. It is possible to compute it over the deseasonalized
    data.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    deseasonal : bool
        Whether to compute the correlation over the 
        deseasonalized data.

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    if (deseasonal) and (season is not None):
        raise ValueError('It is not possible to compute the deseasonal correlation for a seasonal subset')

    if deseasonal:
        target = target[var].load()
        target = xr.apply_ufunc(
            lambda x, mean: x - mean, 
            target.groupby('time.month'),
            target.groupby('time.month').mean()
        ).drop('month')

        pred = pred[var].load()
        pred = xr.apply_ufunc(
            lambda x, mean: x - mean, 
            pred.groupby('time.month'),
            pred.groupby('time.month').mean()
        ).drop('month')

        metric = xr.corr(target, pred, dim='time')
        metric = xr.Dataset({var: metric})

    else:

        metric = xr.corr(target[var], pred[var], dim='time')
        metric = xr.Dataset({var: metric})

    return metric