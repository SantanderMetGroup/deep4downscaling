# SPDX-License-Identifier: MIT

"""
This module contains the building blocks for the DeepESDv2 model.

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

    Maps each coordinate vector to a high-dimensional embedding by evaluating
    sin and cos at geometrically increasing frequencies.  For a scalar input
    *x* and *L* frequency levels the encoding is

    .. math::
        \\gamma(x) = \\bigl[\\sin(2^0 \\pi x),\\; \\cos(2^0 \\pi x),\\;
        \\ldots,\\; \\sin(2^{L-1} \\pi x),\\; \\cos(2^{L-1} \\pi x)\\bigr]

    Applied independently to each of the ``coord_dim`` input dimensions the
    output has ``2 * coord_dim * num_frequencies`` features.  An optional
    learnable linear projection maps this to an arbitrary ``output_dim``.

    Parameters
    ----------
    coord_dim : int
        Dimensionality of the input coordinates (e.g. 2 for lat/lon).

    num_frequencies : int
        Number of frequency octaves *L*.

    output_dim : int or None, optional
        If given, a ``Linear(2 * coord_dim * num_frequencies, output_dim)``
        projection is appended.  When ``None`` the raw Fourier features are
        returned directly.
    """

    def __init__(self, coord_dim, num_frequencies, output_dim=None):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies

        freqs = (2.0 ** torch.arange(num_frequencies)) * math.pi  # (L,)
        self.register_buffer('freqs', freqs)

        raw_dim = 2 * coord_dim * num_frequencies
        self.output_proj = nn.Linear(raw_dim, output_dim) if output_dim is not None else None
        self._output_dim = output_dim if output_dim is not None else raw_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, coords):
        """
        Parameters
        ----------
        coords : Tensor
            Coordinate tensor of shape ``(N, coord_dim)``.

        Returns
        -------
        Tensor
            Positional encoding of shape ``(N, output_dim)`` (or
            ``(N, 2 * coord_dim * num_frequencies)`` when no projection).
        """
        # coords: (N, D), freqs: (L,) -> (N, D, L)
        x = coords.unsqueeze(-1) * self.freqs
        # (N, D, L) -> (N, D*L) interleaved sin/cos -> (N, 2*D*L)
        x = torch.cat([x.sin(), x.cos()], dim=-1)     # (N, D, 2L)
        x = x.reshape(x.shape[0], -1)                  # (N, 2*D*L)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention supporting different query and key/value dimensions.

    The Q projection maps from query_dim to kv_dim, so input queries can have a
    smaller dimension than the encoder tokens. Attention is computed in kv_dim space,
    and the output is also kv_dim-dimensional. This precludes using PyTorch's built-in
    nn.MultiheadAttention, which always outputs at the query input dimension.

    Parameters
    ----------
    query_dim : int
        Dimension of the query input.

    kv_dim : int
        Dimension of the key/value input (encoder tokens). Must be divisible by
        num_heads.

    num_heads : int
        Number of attention heads.

    dropout : float, optional
        Dropout probability on attention weights. Default is 0.0.
    """

    def __init__(self, query_dim, kv_dim, num_heads, dropout=0.):
        super().__init__()
        assert kv_dim % num_heads == 0, 'kv_dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads

        self.q_proj = nn.Linear(query_dim, kv_dim)
        self.k_proj = nn.Linear(kv_dim, kv_dim)
        self.v_proj = nn.Linear(kv_dim, kv_dim)
        self.out_proj = nn.Linear(kv_dim, kv_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, N_q, _ = q.shape

        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N_q, -1)
        return self.out_proj(out)


class MultiHeadLocalCrossAttention(nn.Module):
    """
    Multi-head local cross-attention supporting different query and key/value
    dimensions.

    Each query attends only to a precomputed subset of encoder tokens. The
    locality constraint is enforced through a boolean attention mask rather
    than by gathering per-query key/value slices, so K and V remain shared
    across all queries. This avoids the memory blow-up that a gather-based
    approach causes when the number of queries is much larger than the number
    of encoder tokens (the typical downscaling regime).

    Projections and output space follow the same conventions as
    :class:`MultiHeadCrossAttention`: Q maps from query_dim to kv_dim, and
    attention is computed entirely in kv_dim space.

    Parameters
    ----------
    query_dim : int
        Dimension of the query input.

    kv_dim : int
        Dimension of the key/value input (encoder tokens). Must be divisible by
        num_heads.

    num_heads : int
        Number of attention heads.

    dropout : float, optional
        Dropout probability on attention weights. Default is 0.0.
    """

    def __init__(self, query_dim, kv_dim, num_heads, dropout=0.):
        super().__init__()
        assert kv_dim % num_heads == 0, 'kv_dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads

        self.q_proj = nn.Linear(query_dim, kv_dim)
        self.k_proj = nn.Linear(kv_dim, kv_dim)
        self.v_proj = nn.Linear(kv_dim, kv_dim)
        self.out_proj = nn.Linear(kv_dim, kv_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        """
        Parameters
        ----------
        q : Tensor
            Query tensor of shape ``(B, N_q, query_dim)``.

        k : Tensor
            Key tensor of shape ``(B, N_kv, kv_dim)``.

        v : Tensor
            Value tensor of shape ``(B, N_kv, kv_dim)``.

        attn_mask : Tensor
            Boolean tensor of shape ``(N_q, N_kv)``. Positions that are
            ``True`` are masked out (not attended to).

        Returns
        -------
        Tensor
            Output of shape ``(B, N_q, kv_dim)``.
        """
        B, N_q, _ = q.shape

        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale               # (B, H, N_q, N_kv)
        attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N_q, -1)
        return self.out_proj(out)


class CrossAttentionDecoderBlock(nn.Module):
    """
    Cross-attention decoder block (Perceiver IO style).

    Cross-attention sub-layer with pre-norm on queries and keys/values. Optionally,
    a feedforward MLP sub-layer (also pre-norm via LayerNorm inside the MLP stack)
    with a residual connection. The cross-attention uses a residual connection only
    when ``query_dim == kv_dim`` (subsequent decoder blocks); when
    ``query_dim != kv_dim`` (first block), the dimension change makes a direct
    residual impossible. When enabled, the MLP operates in ``kv_dim`` space.

    Parameters
    ----------
    query_dim : int
        Dimension of the input queries.

    kv_dim : int
        Dimension of the key/value input (encoder tokens).

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Hidden dimension of the feedforward MLP when ``use_cross_attn_mlp`` is
        True.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    use_cross_attn_mlp : bool, optional
        If True, apply a feedforward MLP after cross-attention (with residual).
        Default is False (cross-attention only), matching the original DeepESDv2
        global decoder.
    """

    def __init__(self, query_dim, kv_dim, num_heads, mlp_dim, dropout=0.,
                 use_cross_attn_mlp=False):
        super().__init__()

        self.has_residual = (query_dim == kv_dim)
        self.use_cross_attn_mlp = use_cross_attn_mlp

        self.cross_attn_norm_q = nn.LayerNorm(query_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attention = MultiHeadCrossAttention(query_dim, kv_dim, num_heads, dropout)

        if use_cross_attn_mlp:
            self.mlp = nn.Sequential(nn.LayerNorm(kv_dim),
                                     nn.Linear(kv_dim, mlp_dim),
                                     nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(mlp_dim, kv_dim),
                                     nn.Dropout(dropout))
        else:
            self.mlp = None

    def forward(self, queries, encoder_tokens):
        """
        Parameters
        ----------
        queries : Tensor
            Query tensor of shape ``(B, N_q, query_dim)``.

        encoder_tokens : Tensor
            Encoder tokens of shape ``(B, N_kv, kv_dim)``.

        Returns
        -------
        Tensor
            Output of shape ``(B, N_q, kv_dim)``.
        """
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(encoder_tokens)
        attn_out = self.cross_attention(q, kv, kv)

        if self.has_residual:
            attn_out = queries + attn_out

        if self.use_cross_attn_mlp:
            return attn_out + self.mlp(attn_out)
        return attn_out


class LocalCrossAttentionDecoderBlock(nn.Module):
    """
    Local cross-attention decoder block (Perceiver IO style).

    Same structure as :class:`CrossAttentionDecoderBlock` (optional post-attention
    MLP) but uses :class:`MultiHeadLocalCrossAttention` so that each query only
    attends to a precomputed subset of encoder tokens. This is useful when the
    number of output queries is large and the downscaling task is spatially local.

    Parameters
    ----------
    query_dim : int
        Dimension of the input queries.

    kv_dim : int
        Dimension of the key/value input (encoder tokens).

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Hidden dimension of the feedforward MLP when ``use_cross_attn_mlp`` is
        True.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    use_cross_attn_mlp : bool, optional
        If True, apply a feedforward MLP after cross-attention (with residual).
        Default is True.
    """

    def __init__(self, query_dim, kv_dim, num_heads, mlp_dim, dropout=0.,
                 use_cross_attn_mlp=True):
        super().__init__()

        self.has_residual = (query_dim == kv_dim)
        self.use_cross_attn_mlp = use_cross_attn_mlp

        self.cross_attn_norm_q = nn.LayerNorm(query_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attention = MultiHeadLocalCrossAttention(query_dim, kv_dim, num_heads, dropout)

        if use_cross_attn_mlp:
            self.mlp = nn.Sequential(nn.LayerNorm(kv_dim),
                                     nn.Linear(kv_dim, mlp_dim),
                                     nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(mlp_dim, kv_dim),
                                     nn.Dropout(dropout))
        else:
            self.mlp = None

    def forward(self, queries, encoder_tokens, attn_mask):
        """
        Parameters
        ----------
        queries : Tensor
            Query tensor of shape ``(B, N_q, query_dim)``.

        encoder_tokens : Tensor
            Encoder tokens of shape ``(B, N_kv, kv_dim)``.

        attn_mask : Tensor
            Boolean mask of shape ``(N_q, N_kv)`` (see
            :meth:`MultiHeadLocalCrossAttention.forward`).

        Returns
        -------
        Tensor
            Output of shape ``(B, N_q, kv_dim)``.
        """
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(encoder_tokens)
        attn_out = self.cross_attention(q, kv, kv, attn_mask)

        if self.has_residual:
            attn_out = queries + attn_out

        if self.use_cross_attn_mlp:
            return attn_out + self.mlp(attn_out)
        return attn_out


