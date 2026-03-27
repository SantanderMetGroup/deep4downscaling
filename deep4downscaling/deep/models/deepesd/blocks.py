# SPDX-License-Identifier: MIT

"""
This module contains the building blocks for the NoisyDeepESD model variants.

Authors:
    Jose Gonzalez-Abad
    Carlota Garcia Fernandez
"""

import torch.nn as nn

class ConditionalLayerNorm2d(nn.Module):
    """Conditional layer normalization for 2D feature maps."""

    def __init__(self, dim, noise_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.noise_mlp = nn.Sequential(nn.Conv2d(noise_dim, 2 * dim, kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv2d(2 * dim, 2 * dim, kernel_size=1))

    def forward(self, x, noise):
        # Normalize features across channels
        x = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x)

        # Map noise to gamma and beta
        gamma_beta = self.noise_mlp(noise)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Match dimensions for modulation
        gamma = gamma.permute(0, 2, 3, 1)
        beta = beta.permute(0, 2, 3, 1)

        # Apply conditional modulation
        out = gamma * x_norm + beta

        # Restore channel-first format
        return out.permute(0, 3, 1, 2)
