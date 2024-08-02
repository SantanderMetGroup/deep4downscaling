import torch
from torch.utils.data import Dataset
import numpy as np
import math

class StandardDataset(Dataset):

    """
    Standard dataset for pairs of x and y. The input data must be a
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