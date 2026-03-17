# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains the DeepESDv2 model for statistical downscaling.

DeepESDv2 uses a Vision Transformer (ViT) encoder and a cross-attention decoder.
The decoder builds output queries from Fourier-encoded lat/lon coordinates, plus an
optional learned projection of static covariables (e.g. elevation, land use). Each
query cross-attends to the encoder tokens to produce the prediction at its location.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn

from ..vit.blocks import TransformerBlock
from .blocks import FourierPositionalEncoding, CrossAttentionDecoderBlock


class DeepESDv2(nn.Module):
    """
    DeepESDv2: Vision Transformer encoder with cross-attention decoder for
    statistical downscaling.

    The encoder is identical to ViT: the input is split into patches, embedded,
    processed by transformer blocks, and produces a set of encoder tokens. The
    decoder builds one output query per target location from Fourier-encoded
    lat/lon coordinates (plus optional static covariables), and uses cross-attention
    to gather information from the encoder tokens at each location.

    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must be 4D: (batch, channels, height, width).
        The spatial dimensions must be divisible by patch_size.

    output_coords : torch.Tensor
        Normalized (lat, lon) coordinates of the output locations. Shape (N_out, 2),
        with values in [0, 1]. Registered as a buffer — moves to GPU with the model.

    patch_size : int
        Size of the patches extracted from the input for building token embeddings.
        Must divide both spatial dimensions of the input.

    dim : int
        Embedding dimension for both encoder and decoder. Must be divisible by num_heads.

    depth : int
        Number of transformer encoder blocks.

    num_heads : int
        Number of attention heads in transformer and cross-attention blocks.

    mlp_dim : int
        Hidden dimension of the MLP in transformer and cross-attention blocks.

    decoder_depth : int
        Number of cross-attention decoder blocks.

    num_fourier_freqs : int, optional
        Number of log-spaced frequencies for the Fourier positional encoding of
        lat/lon coordinates. Default is 32.

    use_self_attention : bool, optional
        Whether to apply self-attention among output queries after the cross-attention
        decoder. When enabled, queries are projected to a smaller dimension
        (self_attn_dim) before self-attention to reduce O(N_out^2) cost.
        Default is True.

    self_attn_dim : int, optional
        Intermediate dimension for the post-decoder self-attention. Required when
        use_self_attention is True. Must be divisible by num_heads. Default is None.

    num_vars : int, optional
        Number of output variables. The output shape is always (B, N_out, num_vars).
        Default is 1.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    static_features : torch.Tensor, optional
        Static covariables for each output location, e.g. elevation, land use
        (one-hot encoded), land-sea mask. Shape (N_out, num_static_vars), with all
        features pre-normalized. Registered as a buffer — moves to GPU with the model.
        If provided, a learned Linear(num_static_vars, dim) projection is added to the
        output queries alongside the Fourier positional encoding.

    last_relu : bool, optional
        If True, applies ReLU to the final output. Default is False.
    """

    def __init__(self, x_shape, output_coords, patch_size, dim, depth, num_heads,
                 mlp_dim, decoder_depth, num_fourier_freqs=32, use_self_attention=True,
                 self_attn_dim=None, num_vars=1, dropout=0., static_features=None,
                 last_relu=False):
        super().__init__()

        if len(x_shape) != 4:
            raise ValueError('x_shape must be 4D (B, C, H, W)')

        if x_shape[2] % patch_size != 0 or x_shape[3] % patch_size != 0:
            raise ValueError('Input spatial dimensions must be divisible by patch_size')

        if output_coords.ndim != 2 or output_coords.shape[1] != 2:
            raise ValueError('output_coords must have shape (N_out, 2)')

        if use_self_attention and self_attn_dim is None:
            raise ValueError('self_attn_dim is required when use_self_attention is True')

        if self_attn_dim is not None and self_attn_dim % num_heads != 0:
            raise ValueError('self_attn_dim must be divisible by num_heads')

        self.dim = dim
        self.use_self_attention = use_self_attention
        self.last_relu = last_relu

        # Number of encoder tokens (patches)
        H_tokens = x_shape[2] // patch_size
        W_tokens = x_shape[3] // patch_size
        num_patches = H_tokens * W_tokens

        # Output query coordinates registered as a buffer
        self.register_buffer('output_coords', output_coords)
        N_out = output_coords.shape[0]

        # ENCODER

        self.patch_embedding = nn.Conv2d(x_shape[1], dim,
                                         kernel_size=patch_size, stride=patch_size)
        # Truncated normal init following Dosovitskiy et al. (2020) / Touvron et al. (2021)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout_emb = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(dim)

        # DECODER

        # Fourier encoding of lat/lon coordinates
        self.fourier_enc = FourierPositionalEncoding(num_fourier_freqs, dim, max_log_freq=2.0)

        # Optional learned projection of static covariables
        self.has_static = static_features is not None
        if self.has_static:
            if static_features.ndim != 2 or static_features.shape[0] != N_out:
                raise ValueError('static_features must have shape (N_out, num_static_vars)')
            self.register_buffer('static_features', static_features)
            self.static_embedding = nn.Linear(static_features.shape[1], dim)

        # Truncated normal init following Dosovitskiy et al. (2020) / Touvron et al. (2021)
        self.learned_query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.trunc_normal_(self.learned_query, std=0.02)

        # Cross-attention decoder blocks
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionDecoderBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(dim)

        # Optional self-attention in a reduced dimension for spatial coherence
        if self.use_self_attention:
            self.project_down = nn.Linear(dim, self_attn_dim)
            self.self_attn_norm = nn.LayerNorm(self_attn_dim)
            self.self_attention = nn.MultiheadAttention(self_attn_dim, num_heads,
                                                        dropout=dropout, batch_first=True)
            self.output_head = nn.Linear(self_attn_dim, num_vars)
        else:
            self.output_head = nn.Linear(dim, num_vars)

    def forward(self, x):
        B = x.shape[0]

        # --- ENCODER ---

        # Patch embedding: (B, C, H, W) → (B, dim, H_t, W_t) → (B, N_enc, dim)
        tokens = self.patch_embedding(x)
        tokens = tokens.flatten(2).transpose(1, 2)

        tokens = tokens + self.pos_embedding
        tokens = self.dropout_emb(tokens)

        tokens = self.transformer_blocks(tokens)
        encoder_tokens = self.encoder_norm(tokens)   # (B, N_enc, dim)

        # --- DECODER ---

        # Build output queries from Fourier-encoded coordinates
        query_pos = self.fourier_enc(self.output_coords)   # (N_out, dim)

        # Add static covariable embedding if provided
        if self.has_static:
            query_pos = query_pos + self.static_embedding(self.static_features)

        # Expand to batch and add shared learned content query
        N_out = query_pos.shape[0]
        queries = self.learned_query.expand(B, N_out, -1) + query_pos  # (B, N_out, dim)

        # Cross-attention decoder
        for block in self.decoder_blocks:
            queries = block(queries, encoder_tokens)
        queries = self.decoder_norm(queries)

        # Self-attention in reduced dimension for spatial coherence
        if self.use_self_attention:
            queries = self.project_down(queries)           # (B, N_out, self_attn_dim)
            q = self.self_attn_norm(queries)
            queries = queries + self.self_attention(q, q, q)[0]

        # Output head
        out = self.output_head(queries)   # (B, N_out, num_vars)

        if self.last_relu:
            out = torch.relu(out)

        return out
