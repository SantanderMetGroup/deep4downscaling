import torch
import torch.nn as nn
import numpy as np

class MseLoss(nn.Module):

    """
    Standard Mean Square Error (MSE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            nans_idx = torch.isnan(target[0, :])
            output = output[:, ~nans_idx]
            target = target[:, ~nans_idx]

        loss = torch.mean((target - output) ** 2)
        return loss