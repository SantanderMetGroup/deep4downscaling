# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the building blocks for the DeepESDv2 model.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier positional encoding for multi-dimensional spatial coordinates.

    Maps each coordinate vector (e.g. latitude/longitude of a grid point) to a
    dense positional embedding through a learnable frequency projection followed
    by sinusoidal activations and a linear output projection.

    Li, Y., Si, S., Li, G., Hsieh, C.-J., & Bengio, S. (2021).
    Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding.
    Advances in Neural Information Processing Systems 34 (NeurIPS 2021).
    arXiv:2106.02795.

    Parameters
    ----------
    coord_dim : int
        Dimensionality of the input coordinates (e.g. 2 for latitude/longitude)

    hidden_dim : int
        Number of Fourier features (learnable frequencies).

    output_dim : int
        Dimension of the output positional encoding.
    """

    def __init__(self, coord_dim, hidden_dim, output_dim):
        super().__init__()
        self.freq_proj = nn.Linear(coord_dim, hidden_dim)
        self.output_proj = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, coords):
        """
        Parameters
        ----------
        coords : Tensor
            Coordinate tensor of shape ``(N, coord_dim)``.

        Returns
        -------
        Tensor
            Positional encoding of shape ``(N, output_dim)``.
        """
        x = self.freq_proj(coords)                          # (N, hidden_dim)
        x = torch.cat([x.sin(), x.cos()], dim=-1)          # (N, 2 * hidden_dim)
        return self.output_proj(x)                          # (N, output_dim)


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


class MultiHeadLinearCrossAttention(nn.Module):
    """
    Multi-head linear cross-attention using the approximation from Katharopoulos et al.
    (https://arxiv.org/abs/2006.16236).

    Instead of computing the O(N_q * N_kv) attention matrix, it applies a feature map
    (elu(x) + 1) to queries and keys and computes the attention using the associative
    property of matrix multiplication, reducing the complexity to O(N_q + N_kv).

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
        # Note: Dropout on the attention weights doesn't make as much sense here in the
        # linear formulation, but we can apply it to the output to match the signature.
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, N_q, _ = q.shape

        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply feature map
        q = F.elu_(q) + 1.0
        k = F.elu_(k) + 1.0

        # Compute key-value summary (associativity trick)
        kv_summary = torch.matmul(k.transpose(-2, -1), v)  # (B, H, head_dim, head_dim)

        # Compute denominator
        k_sum = k.sum(dim=-2)  # (B, H, head_dim)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k_sum) + 1e-6)

        # Multiply queries by key-value summary
        out = torch.matmul(q, kv_summary)  # (B, H, N_q, head_dim)

        # Normalize
        out = out * z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().reshape(B, N_q, -1)
        out = self.out_proj(out)
        return self.dropout(out)


class LinearCrossAttentionDecoderBlock(nn.Module):
    """
    Linear cross-attention decoder block (Perceiver IO style).

    Same structure as :class:`CrossAttentionDecoderBlock` (optional post-attention
    MLP) but uses :class:`MultiHeadLinearCrossAttention` to reduce memory and
    compute cost.

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
        self.cross_attention = MultiHeadLinearCrossAttention(query_dim, kv_dim, num_heads, dropout)

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
