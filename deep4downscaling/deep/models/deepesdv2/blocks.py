# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the building blocks for the DeepESDv2 model.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CrossAttentionDecoderBlock(nn.Module):
    """
    Cross-attention decoder block (Perceiver IO style).

    Cross-attention sub-layer followed by a feedforward MLP sub-layer, both with
    pre-norm. The cross-attention uses a residual connection only when query_dim ==
    kv_dim (subsequent decoder blocks); when query_dim != kv_dim (first block), the
    dimension change makes a direct residual impossible. The MLP always operates in
    kv_dim space and always uses a residual connection.

    Parameters
    ----------
    query_dim : int
        Dimension of the input queries.

    kv_dim : int
        Dimension of the key/value input (encoder tokens).

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Hidden dimension of the feedforward MLP.

    dropout : float, optional
        Dropout probability. Default is 0.0.
    """

    def __init__(self, query_dim, kv_dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()

        self.has_residual = (query_dim == kv_dim)

        self.cross_attn_norm_q = nn.LayerNorm(query_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attention = MultiHeadCrossAttention(query_dim, kv_dim, num_heads, dropout)

        self.mlp = nn.Sequential(nn.LayerNorm(kv_dim),
                                 nn.Linear(kv_dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, kv_dim),
                                 nn.Dropout(dropout))

    def forward(self, queries, encoder_tokens):
        q = self.cross_attn_norm_q(queries)
        kv = self.cross_attn_norm_kv(encoder_tokens)
        attn_out = self.cross_attention(q, kv, kv)

        if self.has_residual:
            attn_out = queries + attn_out

        return attn_out + self.mlp(attn_out)
