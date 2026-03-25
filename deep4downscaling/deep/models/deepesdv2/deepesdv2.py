# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the DeepESDv2 model for statistical downscaling.

DeepESDv2 uses a Vision Transformer (ViT) encoder and a cross-attention decoder.
The decoder uses learnable query embeddings (one per output grid point) that
cross-attend to encoder tokens to produce predictions.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn

from ..vit.blocks import TransformerBlock
from .blocks import CrossAttentionDecoderBlock, LocalCrossAttentionDecoderBlock


class DeepESDv2(nn.Module):
    """
    DeepESDv2: Vision Transformer encoder with cross-attention decoder for
    statistical downscaling.

    The encoder is identical to ViT: the input is split into patches, embedded,
    processed by transformer blocks, and produces a set of encoder tokens. The
    decoder uses learnable query embeddings (one per output grid point) of a
    smaller dimension (query_dim) that are projected to the encoder dimension
    through the Q weight matrix of the cross-attention mechanism.

    When ``local_indices`` is provided the decoder uses local cross-attention:
    each query attends only to a precomputed subset of K encoder tokens instead
    of all of them, reducing cost from O(N_out * N_enc) to O(N_out * K).

    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must be 4D: (batch, channels, height, width).
        The spatial dimensions must be divisible by patch_size.

    n_out : int
        Number of output grid points. Determines the number of learnable query
        embeddings.

    patch_size : int
        Size of the patches extracted from the input for building token embeddings.
        Must divide both spatial dimensions of the input.

    dim : int
        Embedding dimension for the encoder. Must be divisible by num_heads.

    depth : int
        Number of transformer encoder blocks.

    num_heads : int
        Number of attention heads in transformer and cross-attention blocks.

    mlp_dim : int
        Hidden dimension of the MLP in the encoder transformer blocks and in the
        decoder cross-attention blocks.

    decoder_depth : int
        Number of cross-attention decoder blocks.

    query_dim : int
        Dimension of the learnable query embeddings. Should be smaller than dim;
        the Q weight matrix in the first cross-attention block maps from query_dim
        to dim.

    local_indices : torch.Tensor or None, optional
        Long tensor of shape ``(n_out, K)`` mapping each query to K encoder
        tokens for local cross-attention. When None (default), standard global
        cross-attention is used.

    num_vars : int, optional
        Number of output variables. The output shape is always (B, N_out, num_vars).
        Default is 1.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    last_relu : bool, optional
        If True, applies ReLU to the final output. Default is False.
    """

    def __init__(self, x_shape, n_out, patch_size, dim, depth, num_heads,
                 mlp_dim, decoder_depth, query_dim, local_indices=None,
                 num_vars=1, dropout=0., last_relu=False):
        super().__init__()

        if len(x_shape) != 4:
            raise ValueError('x_shape must be 4D (B, C, H, W)')

        if x_shape[2] % patch_size != 0 or x_shape[3] % patch_size != 0:
            raise ValueError('Input spatial dimensions must be divisible by patch_size')

        self.last_relu = last_relu

        num_patches = (x_shape[2] // patch_size) * (x_shape[3] // patch_size)

        # ENCODER

        self.patch_embedding = nn.Conv2d(x_shape[1], dim,
                                         kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout_emb = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(dim)

        # DECODER

        self.use_local_attention = local_indices is not None
        if self.use_local_attention:
            self.register_buffer('local_indices', local_indices)
            attn_mask = torch.ones(n_out, num_patches, dtype=torch.bool)
            attn_mask.scatter_(1, local_indices, False)
            self.register_buffer('local_attn_mask', attn_mask)

        self.query_embedding = nn.Parameter(torch.empty(1, n_out, query_dim))
        nn.init.trunc_normal_(self.query_embedding, std=0.02)

        DecoderBlock = LocalCrossAttentionDecoderBlock if self.use_local_attention else CrossAttentionDecoderBlock
        decoder_blocks = []
        for i in range(decoder_depth):
            q_dim = query_dim if i == 0 else dim
            decoder_blocks.append(DecoderBlock(q_dim, dim, num_heads, mlp_dim, dropout))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_norm = nn.LayerNorm(dim)

        self.output_head = nn.Linear(dim, num_vars)

    def forward(self, x):
        B = x.shape[0]

        # ENCODER

        tokens = self.patch_embedding(x)
        tokens = tokens.flatten(2).transpose(1, 2)    # (B, N_enc, dim)

        tokens = tokens + self.pos_embedding
        tokens = self.dropout_emb(tokens)

        tokens = self.transformer_blocks(tokens)
        encoder_tokens = self.encoder_norm(tokens)     # (B, N_enc, dim)

        # DECODER

        queries = self.query_embedding.expand(B, -1, -1)  # (B, N_out, query_dim)

        for block in self.decoder_blocks:
            if self.use_local_attention:
                queries = block(queries, encoder_tokens, self.local_attn_mask)
            else:
                queries = block(queries, encoder_tokens)
        queries = self.decoder_norm(queries)               # (B, N_out, dim)

        out = self.output_head(queries)                    # (B, N_out, num_vars)

        if self.last_relu:
            out = torch.relu(out)

        return out
