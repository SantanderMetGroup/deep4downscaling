# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the NoisyDeepESD architecture for statistical downscaling.

Authors:
    Jose González-Abad
    Carlota García
"""

import torch
from math import ceil
from .blocks import ConditionalLayerNorm2d

class NoisyDeepESD(torch.nn.Module):

    """
    Noisy variant of DeepESD model  that injects noise into the convolutional
    layers to generate stochastic outputs.

    The noise is injected at multiple stages:
    - Before the first convolutional layer: num_channels_noise channels
    - After the first convolutional layer: reduced by 30%
    - After the second convolutional layer: reduced by 60%

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

    num_channels_noise: int
        Number of noise channels to inject into the input. Must be greater than 0.
        The noise is progressively reduced through the network layers.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. By default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, num_channels_noise: int,
                 last_relu: bool=False,
                 members_for_training: int=2):

        super(NoisyDeepESD, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        if num_channels_noise <= 0:
            raise ValueError("num_channels_noise must be greater than 0 for NoisyDeepESD")

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.num_channels_noise = num_channels_noise
        self.last_relu = last_relu
        self.members_for_training = members_for_training

        # First convolutional layer: input channels + noise channels
        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1] + num_channels_noise,
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)
        
        # Second convolutional layer: previous output + reduced noise (70% of original)
        self.num_noise_channels_2 = num_channels_noise - ceil(0.3 * num_channels_noise)
        self.conv_2 = torch.nn.Conv2d(in_channels=50 + self.num_noise_channels_2,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)
        
        # Third convolutional layer: previous output + further reduced noise (40% of original)
        self.num_noise_channels_3 = num_channels_noise - ceil(0.6 * num_channels_noise)
        self.conv_3 = torch.nn.Conv2d(in_channels=25 + self.num_noise_channels_3,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        self.out = torch.nn.Linear(in_features=\
                                   self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                   out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        is_ensemble_mode = self.training or torch.is_grad_enabled()

        if is_ensemble_mode:
            members_to_iterate = self.members_for_training
        else:
            members_to_iterate = 1

        out_members = []
        for i in range(members_to_iterate):
            noise_1 = torch.randn(x.shape[0], self.num_channels_noise, 
                                  x.shape[2], x.shape[3], device=x.device)
            x_feature = torch.cat((x, noise_1), dim=1)

            x_feature = self.conv_1(x_feature)
            x_feature = torch.relu(x_feature)

            noise_2 = torch.randn(x_feature.shape[0], self.num_noise_channels_2, 
                                  x_feature.shape[2], x_feature.shape[3], device=x.device)
            x_feature = torch.cat((x_feature, noise_2), dim=1)

            x_feature = self.conv_2(x_feature)
            x_feature = torch.relu(x_feature)

            noise_3 = torch.randn(x_feature.shape[0], self.num_noise_channels_3, 
                                  x_feature.shape[2], x_feature.shape[3], device=x.device)
            x_feature = torch.cat((x_feature, noise_3), dim=1)

            x_feature = self.conv_3(x_feature)
            x_feature = torch.relu(x_feature)

            x_feature = torch.flatten(x_feature, start_dim=1)

            out_member = self.out(x_feature)
            if self.last_relu: 
                out_member = torch.relu(out_member)
            out_members.append(out_member)
        
        if is_ensemble_mode:
            return out_members
        else:
            return out_members[0]


class NoisyDeepESDCLN(torch.nn.Module):

    """
    Noisy variant of DeepESD model with conditional layer normalization (CLN)
    applied after each convolution and before the activation function.

    The model samples one spatial Gaussian noise tensor per member and uses it
    to condition all CLN layers. Noise is injected only through CLN and is not
    concatenated as channels to the convolutional inputs.

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

    num_channels_noise: int
        Number of channels for the sampled per-member 2D noise map. Must be
        greater than 0.

    noise_dim : int
        Preserved for API compatibility. CLN conditioning now uses the sampled
        noise channels directly (`num_channels_noise`) with no extra embedding.

    sigma : float, optional
        Standard deviation used to scale the sampled Gaussian noise before CLN.
        By default is set to 1.0.

    last_relu: bool, optional
        If set to True, the output of the last dense layer is passed through a
        ReLU activation function. By default is set to False.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, num_channels_noise: int,
                 noise_dim: int=32, last_relu: bool=False,
                 members_for_training: int=2, sigma: float=1.0):

        super(NoisyDeepESDCLN, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        if num_channels_noise <= 0:
            raise ValueError("num_channels_noise must be greater than 0 for NoisyDeepESDCLN")

        if noise_dim <= 0:
            raise ValueError("noise_dim must be greater than 0 for NoisyDeepESDCLN")
        if sigma < 0:
            raise ValueError("sigma must be non-negative for NoisyDeepESDCLN")

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.num_channels_noise = num_channels_noise
        self.noise_dim = noise_dim
        self.last_relu = last_relu
        self.members_for_training = members_for_training
        self.sigma = sigma

        # First convolutional layer
        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)
        self.cln_1 = ConditionalLayerNorm2d(50, num_channels_noise)

        # Second convolutional layer
        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)
        self.cln_2 = ConditionalLayerNorm2d(25, num_channels_noise)

        # Third convolutional layer
        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)
        self.cln_3 = ConditionalLayerNorm2d(self.filters_last_conv, num_channels_noise)

        self.out = torch.nn.Linear(in_features=\
                                   self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                   out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        is_ensemble_mode = self.training or torch.is_grad_enabled()

        if is_ensemble_mode:
            members_to_iterate = self.members_for_training
        else:
            members_to_iterate = 1

        out_members = []
        for i in range(members_to_iterate):
            eps = torch.randn(x.shape[0], self.num_channels_noise,
                              x.shape[2], x.shape[3], device=x.device) * self.sigma

            x_feature = self.conv_1(x)
            x_feature = self.cln_1(x_feature, eps)
            x_feature = torch.relu(x_feature)

            x_feature = self.conv_2(x_feature)
            x_feature = self.cln_2(x_feature, eps)
            x_feature = torch.relu(x_feature)

            x_feature = self.conv_3(x_feature)
            x_feature = self.cln_3(x_feature, eps)
            x_feature = torch.relu(x_feature)

            x_feature = torch.flatten(x_feature, start_dim=1)

            out_member = self.out(x_feature)
            if self.last_relu:
                out_member = torch.relu(out_member)
            out_members.append(out_member)

        if is_ensemble_mode:
            return out_members
        else:
            return out_members[0]

