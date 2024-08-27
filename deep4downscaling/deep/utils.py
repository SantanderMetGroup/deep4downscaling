import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import math

class StandardDataset(Dataset):

    """
    Standard Pytorch dataset for pairs of x and y. The input data must be a
    np.ndarray.

    Parameters
    ----------
    x : np.ndarray
        Array representing the predictor data

    y : np.ndarray
        Array representing the predictand data
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = self.x[idx, :]
        y = self.y[idx, :]
        return x, y

class MultivariateDataset(Dataset):

    """
    Pytorch dataset for pairs of x and multiple y(s). The input data must be a
    np.ndarray. This Dataset returns the predictors as the first element and the
    multiple target as subsequent elements.

    Parameters
    ----------
    x : np.ndarray
        Array representing the predictor data

    y : np.ndarray
        Arrays representing the predictand data
    """

    def __init__(self, x: np.ndarray, *y: np.ndarray) -> None:
        self.x = torch.tensor(x)

        self.y_list = []
        for elem in y: 
            self.y_list.append(torch.tensor(elem))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = self.x[idx, :]

        y_list = []
        for elem in self.y_list:
            y_list.append(elem[idx, :])

        return x, *y_list

def precipitation_NLL_trans(data: xr.Dataset, threshold: float) -> xr.Dataset:
    
    """
    This function performs the transformation required for training
    a model with the NLL of a Bernoulli and gamma distributions. The
    main idea is to set a threshold that defines the wet days, so the
    DL model learns the gamma only on wet days, avoiding biased amounts
    if including the amount for dry days.

    Parameters
    ----------
    data : xr.Dataset
        Data to apply the transformation to.

    threshold : float
        Threshold defining the amount for wet days.

    Returns
    -------
    xr.Dataset
        The transformed data
    """

    data_final = data.copy(deep=True)

    epsilon = 1e-06
    threshold = threshold - epsilon # Include in the distribution of wet days the threshold value
    data_final = data_final - threshold
    data_final = xr.where(cond=data_final<0, x=0, y=data_final)

    return data_final