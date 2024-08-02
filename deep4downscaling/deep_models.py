import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DeepESDtas(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024. Thi implementation
    allows for a deterministic (MSE-based) and stochastic (NLL-based) definition.

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

    stochastic: bool
        If set to True, the model is composed of two final dense layers computing
        the mean and log fo the variance. Otherwise, the models is composed of one
        final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDtas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic

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

        if self.stochastic:
            self.out_mean = torch.nn.Linear(in_features=\
                                                self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.out_log_var = torch.nn.Linear(in_features=\
                                                self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                               out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            mean = self.out_mean(x)
            log_var = self.out_log_var(x)
            out = torch.cat((mean, log_var), dim=1)
        else:
            out = self.out(x)

        return out