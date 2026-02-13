# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the DeepESD discriminator (cGAN) for statistical downscaling.

Authors:
    Alfonso Hernanz
    Jose González-Abad
"""

import torch

class DeepESD_Discriminator(torch.nn.Module):
    """
    Discriminator for the DeepESD-based cGAN.
    Receives (X, Y) pairs and predicts the probability of Y being real.

    X: (batch, channels, lat, lon)
    Y: (batch, stations)
    Output: scalar in [0,1] per sample.

    Architecture:
      - CNN encoder for X
      - Concatenate flattened X features + Y
      - Fully-connected layers for classification

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
    """

    def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int = 25):
        super(DeepESD_Discriminator, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            raise ValueError("X must have 4 dims and Y must have 2 dims.")

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

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

        n_features_x = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv
        n_features_total = n_features_x + self.y_shape[1]

        self.fc1 = torch.nn.Linear(n_features_total, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Predictor tensor (batch, channels, lat, lon)
        y : torch.Tensor
            Predictand tensor (batch, stations)

        Returns
        -------
        torch.Tensor
            Probability that (x, y) is real, shape (batch, 1)
        """
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_3(x))

        x = torch.flatten(x, start_dim=1)

        xy = torch.cat((x, y), dim=1)

        xy = self.leaky_relu(self.fc1(xy))
        xy = self.leaky_relu(self.fc2(xy))
        out = self.sigmoid(self.fc3(xy))

        return out

