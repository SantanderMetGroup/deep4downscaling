import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DeepESDMultiHead(torch.nn.Module):

    """
    Multi-head version of DeepESD for statistical downscaling with ensemble predictions.
    It is possible to compute this model with multiple output heads.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    num_heads: int
        Number of output heads for the multi-head architecture.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. This does not apply when stochastic=True. By
        default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, num_heads: int,
                 last_relu: bool=False):

        super(DeepESDMultiHead, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.num_heads = num_heads
        self.last_relu = last_relu

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        self.out_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(in_features=self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                out_features=self.y_shape[1])
            )
            for _ in range(self.num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        out = [head(x) for head in self.out_heads]

        if self.last_relu: 
            out = [torch.relu(element) for element in out]

        return out