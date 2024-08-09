import xarray as xr
import numpy as np

def remove_days_with_nans(data: xr.Dataset,
                          coord_names: dict={'lat': 'lat',
                                             'lon': 'lon'}) -> xr.Dataset:

    """
    Remove the days with at least one nan across the spatial domain. This function
    applies to all the variables composing the data

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to filter

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    xr.Dataset
        Filtered xarray Dataset
    """

    # Get time indices with zero null values
    nans_indices = data.isnull()
    nans_indices = nans_indices.sum(dim=(coord_names['lat'],
                                         coord_names['lon'])).to_array().values
    nans_indices = np.logical_or.reduce(nans_indices, axis=0)
    nans_indices = ~nans_indices

    # Filter the Dataset
    data = data.sel(time=nans_indices)

    # Log the operation
    if np.sum(nans_indices) == len(nans_indices):
        print('There are no observations containing null values')
    else:
        print(f'Removing {np.sum(nans_indices)} observations contaning null values')

    return data

def align_datasets(data_1: xr.Dataset, data_2: xr.Dataset, coord: str) -> (xr.Dataset, xr.Dataset):

    """
    Align two Datasets with respect to the coord

    Parameters
    ----------
    data_1, data_2 : xr.Dataset, xr.Dataset
        Xarray datasets to align

    coord : str
        Coordinate to use to align both Datasets

    Returns
    -------
    xr.Dataset, xr.Dataset) 
        Aligned Datasets      
    """

    data_1 = data_1.sel(time=np.in1d(data_1[coord].values,
                                     data_2[coord].values))
    data_2 = data_2.sel(time=np.in1d(data_2[coord].values,
                                     data_1[coord].values))

    return data_1, data_2

def standardize(data_ref: xr.Dataset, data: xr.Dataset) -> xr.Dataset:

    """
    Standardize the data with the mean and std computed over the data_ref.
    x' = (x - mean) / std

    Parameters
    ----------
    data_ref : xr.Dataset
        Data used as reference to compute the mean and standard deviaton

    data : xr.DataSet
        Data to standardize

    Returns
    -------
    xr.Dataset        
        Standardize data
    """

    mean = data_ref.mean('time')
    std = data_ref.std('time')

    data_stand = (data - mean) / std

    return data_stand

def xarray_to_numpy(data: xr.Dataset) -> np.ndarray:

    """
    Converts a xr.Dataset to np.ndarray by relying on to_numpy()
    from xarray.

    Parameters
    ----------
    data : xr.Dataset
        Data to convert

    Returns
    -------
    np.ndarray
        data converted to numpy
    """

    final_data = []
    data_vars = [i for i in data.data_vars]
    for var_convert in data_vars:
        final_data.append(data[var_convert].to_numpy())

    if len(data_vars) == 1:
        final_data = final_data[0]
    else:
        final_data = np.stack(final_data, axis=1)

    return final_data

def compute_valid_mask(data: xr.Dataset) -> xr.Dataset:

    """
    Compute a mask indicating whether for each spatial point there is
    any nan (0) or not (1). This function collapses the time dimension.

    Parameters
    ----------
    data : xr.Dataset
        Data used to compute the mask

    Returns
    -------
    xr.Dataset
        Mask with 1 for spatial locations with non-nans and 0 otherwise
    """

    data_mask = data.isnull().astype('int').mean('time')
    data_mask = xr.where(data_mask == 0, 1, 0)

    return data_mask

def split_data(*data: np.ndarray,
               split_percentage: float, shuffle: bool, seed: int=None) -> np.ndarray:

    """
    Split the input data into two new datasets in the first dimension of the
    data, which in our context generally corresponds to time. This split is
    performed based on the value provided through the split_percentage argument.

    Parameters
    ----------
    data : np.ndarray
        Data to split. It accepts any number of np.ndarray objects. It is
        required for these to have the same first dimension, otherwise this
        function will throw an error.
    
    split_percentage : float
        What percentage of data to reserve for the new dataset. For instance,
        a value of 0.1 corresponds to the 10%.

    shuffle : bool
        Whether to shuffle the data before splitting or not.

    seed : int
        If provided the shuffling is done taking into account this numpy
        seed

    Note
    ----
    The function torch.utils.data.random_split does the same but over
    torch.Dataset objects.

    Returns
    -------
    np.ndarray
        Returns the splitted data. This can be seen as two sets, the first corresponds
        to the np.ndarray provided in the data argument (first split) and the second with
        the second split.
    """

    if seed is not None:
        np.random.seed(seed)

    lens_data = [x.shape[0] for x in data]
    if len(set(lens_data)) != 1:
        error_msg =\
        'All data provided must have the same number of elements across'
        'the first dimension'
        
        raise ValueError(error_msg)

    idxs = list(range(data[0].shape[0]))

    if shuffle:
        np.random.shuffle(idxs)
    
    # This copy is needed for the reordering to have effect
    data_copy = []
    for x in data:
        data_copy.append(np.array(x[idxs, :], copy=True))

    split_threshold = round((1-split_percentage) * len(idxs))

    data_split_1 = []
    for x in data_copy:
        data_split_1.append(x[:split_threshold, :])

    data_split_2 = []
    for x in data_copy:
        data_split_2.append(x[split_threshold:, :])

    return (*data_split_1, *data_split_2)

def scaling_delta_correction(data: xr.Dataset,
                             gcm_hist: xr.Dataset, obs_hist: xr.Dataset) -> xr.Dataset:

    """
    Apply the scaling delta correction proposed in Baño-Medina et al. 2024 to bias-correct
    the mean and standard deviation of data (GCM in historical/future scenrios) to "take it 
    closer" to the distribution of the observations used to train the DL model (obs_hist).
    When bias-correcting a GCM in a future scenario, the climate change seasonal is not lost,
    as a delta between the future and historical climatology is added back to the bias-corrected
    data (see Baño-Medina et al. 2024 or González-Abad 2024 for more details). By default this
    correction is performed in a monthly basis.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758.

    González Abad, J. (2024). Towards explainable and physically-based deep learning
    statistical downscaling methods. PhD thesis, Universidad de Cantabria.

    Parameters
    ----------
    data : xr.Dataset
        Data to bias-correct. This should correspond to the GCM in any historical
        or future scenario.

    gcm_hist : xr.Dataset
        Historical period of the GCM to take as reference.

    obs_hist : xr.Dataset
        Observational period to take as reference.

    Returns
    -------
    xr.Dataset
        The bias-corrected data.
    """

    # Make a copy for the final object
    data_final = data.copy(deep=True)

    # Define the scaling delta correction
    def _correction(data: xr.Dataset,
                    gcm_hist: xr.Dataset, obs_hist: xr.Dataset) -> xr.Dataset:

        data_final = data.copy(deep=True)

        data_mean = data.mean('time')
        gcm_hist_mean = gcm_hist.mean('time')
        obs_hist_mean = obs_hist.mean('time')

        gcm_hist_sd = gcm_hist.std('time')
        obs_hist_sd = obs_hist.std('time')

        delta = data_mean - gcm_hist_mean

        data_final = ((((data_final - delta - gcm_hist_mean)/gcm_hist_sd) * \
                   obs_hist_sd) + obs_hist_mean + delta)

        return data_final

    # Apply the correction by month
    months = np.unique(data['time.month'].values)
    for month in months:

        data_month = data.sel(time=data.time.dt.month.isin(month))
        gcm_hist_month = gcm_hist.sel(time=gcm_hist.time.dt.month.isin(month))
        obs_hist_month = obs_hist.sel(time=obs_hist.time.dt.month.isin(month))

        data_corrected = _correction(data=data_month,
                                     gcm_hist=gcm_hist_month,
                                     obs_hist=obs_hist_month)

        # Assign the corrected variable to data_final
        for var in data_final.keys():
            data_final[var][data_final.time.dt.month.isin(month)] = data_corrected[var]

    return data_final