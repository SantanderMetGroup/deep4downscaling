# SPDX-License-Identifier: MIT

"""
Building blocks of the DeepESDv2 model.

Authors:
    Jose González-Abad
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFPositionalEncoding(nn.Module):
    """
    Multi-frequency sinusoidal positional encoding (Mildenhall et al., ECCV 2020).

    Each coordinate is mapped to sin/cos at geometrically increasing
    frequencies; for a scalar input *x* and *L* frequency levels

    .. math::
        \\gamma(x) = \\bigl[\\sin(2^0 \\pi x),\\; \\cos(2^0 \\pi x),\\;
        \\ldots,\\; \\sin(2^{L-1} \\pi x),\\; \\cos(2^{L-1} \\pi x)\\bigr]

    Applied to each of the ``coord_dim`` input dimensions this gives
    ``2 * coord_dim * num_frequencies`` features, optionally projected to
    ``output_dim`` by a learned linear layer. Coordinates are expected in
    roughly ``[-1, 1]``: the shortest wavelength is then ``2 / 2^(L-1)``.

    Parameters
    ----------
    coord_dim : int
        Dimensionality of the coordinates (2 for lat/lon).

    num_frequencies : int
        Number of frequency octaves *L*.

    output_dim : int or None, optional
        Size of the learned output projection. When None, the raw Fourier
        features are returned.
    """

    def __init__(self, coord_dim, num_frequencies, output_dim=None):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies

        freqs = (2.0 ** torch.arange(num_frequencies)) * math.pi   # (L,)
        self.register_buffer('freqs', freqs)

        raw_dim = 2 * coord_dim * num_frequencies
        self.output_proj = nn.Linear(raw_dim, output_dim) if output_dim is not None else None
        self._output_dim = output_dim if output_dim is not None else raw_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, coords):
        """``coords``: (N, coord_dim) -> (N, output_dim)."""
        x = coords.unsqueeze(-1) * self.freqs                      # (N, D, L)
        x = torch.cat([x.sin(), x.cos()], dim=-1)                  # (N, D, 2L)
        x = x.reshape(x.shape[0], -1)                              # (N, 2DL)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


class ConvStem(nn.Module):
    """
    Convolutional stem: two 3x3 convolutions that turn every predictor grid
    point into a ``dim``-dimensional token.

    Compared with a 1x1 (per-point linear) embedding, the 3x3 neighbourhood
    lets each token carry local horizontal structure of the fields (gradients,
    curvature: e.g. vorticity, convergence, advection) before self-attention.
    Early convolutions also make ViT optimisation easier (Xiao et al., 2021).

    Parameters
    ----------
    in_channels : int
        Number of predictor variables.

    dim : int
        Token dimension.
    """

    def __init__(self, in_channels, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1))

    def forward(self, x):
        """``x``: (B, C, H, W) -> (B, dim, H, W)."""
        return self.net(x)


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention from decoder queries to encoder tokens.

    Uses ``F.scaled_dot_product_attention`` (FlashAttention / memory-efficient
    kernels), so the (N_q, N_kv) attention matrix is never materialised.
    The output projection is zero-initialised: with the residual connection of
    the surrounding block, each decoder block starts as the identity
    (DETR / Perceiver IO).

    Parameters
    ----------
    dim : int
        Dimension of queries and tokens. Must be divisible by ``num_heads``.

    num_heads : int
        Number of attention heads.

    dropout : float, optional
        Dropout on the attention weights (training only). Default 0.
    """

    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, queries, tokens):
        """``queries``: (B, N_q, dim), ``tokens``: (B, N_kv, dim) -> (B, N_q, dim)."""
        B, N_q, _ = queries.shape
        q = self.q_proj(queries).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)   # (B, H, N_q, hd)
        return self.out_proj(out.transpose(1, 2).reshape(B, N_q, -1))


class CrossAttentionDecoderBlock(nn.Module):
    """
    Decoder block (Perceiver IO style): pre-norm cross-attention from the
    queries to the encoder tokens, then a pre-norm MLP, each with a residual
    connection. Queries never attend to each other, so every output point is
    decoded independently given the encoder tokens.

    Parameters
    ----------
    dim : int
        Dimension of queries and tokens.

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Hidden size of the MLP.

    dropout : float, optional
        Dropout probability. Default 0.
    """

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, queries, tokens):
        """``queries``: (B, N_q, dim), ``tokens``: (B, N_kv, dim) -> (B, N_q, dim)."""
        queries = queries + self.attn(self.norm_q(queries), self.norm_kv(tokens))
        return queries + self.mlp(queries)
