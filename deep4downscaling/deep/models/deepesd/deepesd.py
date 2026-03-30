# SPDX-License-Identifier: MIT

"""
This module contains the DeepESD architecture for statistical downscaling.

Authors:
    Jose González-Abad
"""

from __future__ import annotations
import warnings
from typing import Literal

import torch

Predictand = Literal["tas", "pr"]

class DeepESD(torch.nn.Module):
    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for statistical
    downscaling (temperature or precipitation). Supports deterministic
    (MSE-based) and stochastic (NLL-based) heads depending on predictand.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

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

    stochastic : bool
        If True, the final layers follow the probabilistic formulation for the
        chosen predictand (Gaussian for tas, Bernoulli-gamma for pr).
        If False, a single linear maps to the predictand gridpoints.

    predictand : 'tas' or 'pr'
        tas: mean and log-variance heads when stochastic.
        pr: p, log-shape and log-scale when stochastic

    last_relu : bool, optional
        If True, apply ReLU to the deterministic output. Default False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool,
                 predictand: Predictand, last_relu: bool = False):
        super().__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            raise ValueError("X and Y data must have dimensions of length 4 and 2, respectively")

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic
        self.predictand = predictand
        self.last_relu = last_relu if predictand == "pr" else False

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50, kernel_size=3, padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25, out_channels=self.filters_last_conv,
                                      kernel_size=3, padding=1)

        flat_in = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv
        n_out = self.y_shape[1]

        if self.stochastic:
            if predictand == "tas":
                self.out_mean = torch.nn.Linear(in_features=flat_in, out_features=n_out)
                self.out_log_var = torch.nn.Linear(in_features=flat_in, out_features=n_out)
            elif predictand == "pr":
                self.p = torch.nn.Linear(in_features=flat_in, out_features=n_out)
                self.log_shape = torch.nn.Linear(in_features=flat_in, out_features=n_out)
                self.log_scale = torch.nn.Linear(in_features=flat_in, out_features=n_out)
            else:
                raise ValueError(f"Invalid predictand: {predictand}")
        else:
            self.out = torch.nn.Linear(in_features=flat_in, out_features=n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            if self.predictand == "tas":
                mean = self.out_mean(x)
                log_var = self.out_log_var(x)
                out = torch.cat((mean, log_var), dim=1)
            elif self.predictand == "pr":
                p = self.p(x)
                p = torch.sigmoid(p)
                log_shape = self.log_shape(x)
                log_scale = self.log_scale(x)
                out = torch.cat((p, log_shape, log_scale), dim=1)
            else:
                raise ValueError(f"Invalid predictand: {self.predictand}")
        else:
            out = self.out(x)
            if self.last_relu:
                out = torch.relu(out)

        return out


class DeepESDtas(DeepESD):
    """
    DeepESD for temperature downscaling. Equivalent to
    DeepESD(..., predictand='tas'). Deprecated.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

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
        the mean and log of the variance. Otherwise, the model is composed of one
        final layer computing the values.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):
        warnings.warn("DeepESDtas is deprecated; use DeepESD(..., predictand='tas') instead.",
                      FutureWarning, stacklevel=2)
        super().__init__(x_shape=x_shape, y_shape=y_shape, filters_last_conv=filters_last_conv,
                         stochastic=stochastic, predictand="tas")


class DeepESDpr(DeepESD):
    """
    DeepESD for precipitation downscaling. Equivalent to
    DeepESD(..., predictand='pr'). Deprecated.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

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
        If set to True, the model is composed of three final dense layers computing
        the p, shape and scale of the Bernoulli-gamma distribution. Otherwise,
        the model is composed of one final layer computing the values.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. This does not apply when stochastic=True. By
        default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int,
                 stochastic: bool, last_relu: bool = False):
        warnings.warn("DeepESDpr is deprecated; use DeepESD(..., predictand='pr') instead.",
                      FutureWarning, stacklevel=2)
        super().__init__(x_shape=x_shape, y_shape=y_shape, filters_last_conv=filters_last_conv,
                         stochastic=stochastic, predictand="pr", last_relu=last_relu)
