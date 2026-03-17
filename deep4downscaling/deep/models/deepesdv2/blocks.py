# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the building blocks for the DeepESDv2 model.

Authors:
    Jose González-Abad
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 2D spatial coordinates (lat, lon).

    Maps normalized (lat, lon) coordinates to a high-dimensional vector using
    log-spaced sinusoidal features followed by a learned linear projection. The
    multi-scale sinusoidal basis lets the model distinguish fine spatial differences
    (high frequencies) while still capturing large-scale structure (low frequencies).

    Based on Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains".

    Parameters
    ----------
    num_frequencies : int
        Number of log-spaced frequencies for the sinusoidal encoding.

    dim : int
        Output embedding dimension.

    max_log_freq : float, optional
        Maximum log10 frequency. Frequencies are spaced from 10^0 to 10^max_log_freq.
        Default is 4.0.
    """

    def __init__(self, num_frequencies, dim, max_log_freq=4.):
        super().__init__()
        # Fixed log-spaced frequencies (not learnable)
        # Using a smaller max_log_freq to prevent high-frequency "checkerboard" artifacts
        # commonly seen in implicitly neural representations and Fourier features
        max_log_freq = min(max_log_freq, 2.0)  # Capped at 2.0
        self.register_buffer('freqs', torch.logspace(0, max_log_freq, num_frequencies))
        # Learned projection: (2 coords) * (num_frequencies) * (sin + cos) → dim
        self.proj = nn.Linear(2 * num_frequencies * 2, dim)

    def forward(self, coords):
        # coords: (N, 2) — normalized lat/lon in [0, 1]
        x = coords.unsqueeze(-1) * self.freqs           # (N, 2, F)
        x = torch.cat([torch.sin(x), torch.cos(x)], -1) # (N, 2, 2F)
        x = x.flatten(1)                                 # (N, 4F)
        return self.proj(x)                              # (N, dim)


class CrossAttentionDecoderBlock(nn.Module):
    """
    Cross-attention decoder block.

    Each block has two sub-layers (all with pre-norm and residual connections):
      1. Cross-attention: output queries attend to encoder tokens (Q=queries, K/V=encoder).
      2. MLP: non-linear transformation of the gathered features.

    Parameters
    ----------
    dim : int
        Embedding dimension.

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Hidden dimension of the MLP.

    dropout : float, optional
        Dropout probability. Default is 0.0.
    """

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()

        # Cross-attention: Q from queries, K/V from encoder tokens
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads,
                                                     dropout=dropout, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, queries, encoder_tokens):
        # queries: (B, N_out, dim)
        # encoder_tokens: (B, N_enc, dim)

        # Cross-attention: each output query gathers information from encoder tokens
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(encoder_tokens)
        queries = queries + self.cross_attention(q, kv, kv)[0]

        # MLP
        queries = queries + self.mlp(queries)

        return queries