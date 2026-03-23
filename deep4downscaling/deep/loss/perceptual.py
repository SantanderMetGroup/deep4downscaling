# SPDX-License-Identifier: GPL-3.0-or-later

"""
Perceptual loss function using VGG19 features.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class PerceptualLoss(nn.Module):

    """
    Perceptual loss function using VGG19 features. This loss combines
    a perceptual term based on VGG feature differences with an MSE term.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    device : str
        Device to run the VGG model on ('cuda' or 'cpu').

    perceptual_loss_type : str
        Type of loss to compute on VGG features. Either 'l1' or 'mse'.

    layer_weights : dict
        Dictionary mapping layer names to their indices for feature extraction.

    reshape_dim : int
        Spatial dimension to reshape 2D inputs to (assumes square images).

    perceptual_term_weight : float, optional
        Weight for the perceptual loss term. Default: 1.0.

    mse_term_weight : float, optional
        Weight for the MSE loss term. Default: 1.0.

    target : torch.Tensor
        Target/ground-truth data.

    output : torch.Tensor
        Predicted data (model's output).
    """

    def __init__(self, ignore_nans: bool, device: str, perceptual_loss_type: str,
                 layer_weights: dict, reshape_dim: int,
                 perceptual_term_weight: float = 1.0,
                 mse_term_weight: float = 1.0) -> None:
        super(PerceptualLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.device = device
        self.perceptual_loss_type = perceptual_loss_type.lower()
        if self.perceptual_loss_type not in ['l1', 'mse']:
            raise ValueError("perceptual_loss_type must be 'l1' or 'mse'")
        self.reshape_dim = reshape_dim
        self.perceptual_term_weight = perceptual_term_weight
        self.mse_term_weight = mse_term_weight

        self.vgg = VGG19Features(layer_weights=layer_weights).to(device)
        self.vgg.eval()

        # Disable gradients for VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        if self.perceptual_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input for VGG: reshape to 4D and normalize.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, can be 2D (flattened) or 4D.

        Returns
        -------
        torch.Tensor
            Normalized 4D tensor ready for VGG.
        """
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, self.reshape_dim, self.reshape_dim))
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.normalize(x)

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            if torch.isnan(target).any() or torch.isnan(output).any():
                print("TODO: NanS handling")

        # Preprocess inputs for VGG
        target_p = self._preprocess_input(target)
        output_p = self._preprocess_input(output)

        # Get the VGG features
        target_features = self.vgg(target_p)
        output_features = self.vgg(output_p)

        # Compute perceptual loss
        perceptual_loss = 0.0
        for i in range(len(target_features)):
            perceptual_loss += self.criterion(output_features[i], target_features[i])

        # Compute MSE loss
        mse_loss = torch.mean((target - output) ** 2)

        return self.perceptual_term_weight * perceptual_loss + self.mse_term_weight * mse_loss


class VGG19Features(nn.Module):

    """
    VGG19 feature extractor for perceptual loss.

    Extracts features from specified layers of a pretrained VGG19 network.

    Parameters
    ----------
    layer_weights : dict, optional
        Dictionary mapping layer names to their indices.
        If None, default layers (conv2_1, conv3_1) are used.
    """

    def __init__(self, layer_weights: dict = None) -> None:
        super(VGG19Features, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        default_layers = {
            # 'conv1_1': 0,
            'conv2_1': 5,
            'conv3_1': 10,
            # 'conv4_1': 19,
            # 'conv5_1': 28
        }

        self.selected_layers_indices = sorted(list(default_layers.values()))

        self.features = nn.Sequential(*list(vgg19.features.children())[:self.selected_layers_indices[-1] + 1])

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        """
        Extract features from selected VGG layers.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input tensor.

        Returns
        -------
        list
            List of feature maps from selected layers.
        """
        output_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.selected_layers_indices:
                output_features.append(x)
        return output_features
