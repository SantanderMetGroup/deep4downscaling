# SPDX-License-Identifier: MIT

"""
This module contains the DeepESDv2 model for statistical downscaling.

DeepESDv2 uses a Vision Transformer (ViT) encoder and a cross-attention decoder.
The decoder queries can be learnable embeddings or NeRF positional encodings
derived from the geographic coordinates of each output grid point.

Authors:
    Jose González-Abad
"""

from typing import Optional

import torch
import torch.nn as nn

from ..vit.blocks import TransformerBlock
from .blocks import (CrossAttentionDecoderBlock, LocalCrossAttentionDecoderBlock,
                      NeRFPositionalEncoding)


class DeepESDv2(nn.Module):
    """
    DeepESDv2: Vision Transformer encoder with cross-attention decoder for
    statistical downscaling.

    The encoder is identical to ViT: the input is split into patches, embedded,
    processed by transformer blocks, and produces a set of encoder tokens. The
    decoder uses query embeddings of dimension ``query_dim`` that are projected
    to the encoder dimension through the Q weight matrix of the cross-attention
    mechanism. Two query-embedding modes are supported:

    * **Learnable (default):** one free ``nn.Parameter`` per output grid point,
      used when ``grid_coords`` is ``None``.
    * **NeRF positional encoding:** multi-frequency sinusoidal encoding
      (Mildenhall et al., ECCV 2020) that evaluates sin/cos at geometrically
      increasing frequencies, producing highly discriminative embeddings even
      for spatially close grid points. Activated by passing ``grid_coords``.
      A linear projection maps the raw features to ``query_dim``.

    When ``local_indices`` is provided the decoder uses local cross-attention:
    each query attends only to a precomputed subset of K encoder tokens instead
    of all of them, reducing cost from O(N_out * N_enc) to O(N_out * K).

    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must be 4D: (batch, channels, height, width).
        The spatial dimensions must be divisible by patch_size.

    n_out : int
        Number of output grid points.

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
        Hidden dimension of the MLP in the encoder transformer blocks and, when
        the decoder applies a post-cross-attention MLP, of that MLP as well.

    decoder_depth : int
        Number of cross-attention decoder blocks.

    query_dim : int
        Dimension of the decoder query embeddings. Used by the learnable mode
        and as the output projection dimension in the NeRF mode.

    local_indices : torch.Tensor or None, optional
        Long tensor of shape ``(n_out, K)`` mapping each query to K encoder
        tokens for local cross-attention. When None (default), standard global
        cross-attention is used.

    decoder_cross_attn_mlp : bool or None, optional
        Whether each decoder block applies a feedforward MLP after cross-attention.
        If ``None`` (default), the global cross-attention decoder omits this MLP
        (historical behavior), while local decoders include it. If ``True`` or
        ``False``, that choice applies to all decoder block types.

    grid_coords : torch.Tensor or None, optional
        Float tensor of shape ``(n_out, coord_dim)`` with the spatial coordinates
        of each output grid point. Coordinates should be pre-normalized to a
        suitable range (e.g. ``[-1, 1]``) before being passed; the model stores
        and uses them as-is. When provided, NeRF positional encoding is used for
        the decoder queries; when ``None`` (default), learnable embeddings are
        used instead.

    nerf_num_frequencies : int, optional
        Number of frequency octaves for the NeRF encoding. Only used when
        ``grid_coords`` or ``encoder_grid_coords`` is provided. Default is 10.

    encoder_grid_coords : torch.Tensor or None, optional
        Float tensor of shape ``(num_patches, coord_dim)`` with the spatial
        coordinates of each encoder patch center. When provided, NeRF
        positional encoding replaces the learnable encoder position
        embeddings. When ``None`` (default), learnable embeddings are used.

    share_nerf_encoding : bool, optional
        If True and both ``grid_coords`` and ``encoder_grid_coords`` are
        provided, the encoder and decoder share the same
        ``NeRFPositionalEncoding`` module (including its learnable linear
        projection). This requires ``dim == query_dim``. Default is False.

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
                 decoder_cross_attn_mlp: Optional[bool] = None,
                 grid_coords=None, nerf_num_frequencies=10,
                 encoder_grid_coords=None, share_nerf_encoding=False,
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

        self.share_nerf_encoding = share_nerf_encoding and encoder_grid_coords is not None

        if encoder_grid_coords is not None:
            if encoder_grid_coords.shape[0] != num_patches:
                raise ValueError('encoder_grid_coords first dim must equal num_patches')
            self.register_buffer('encoder_grid_coords', encoder_grid_coords)
            self.pos_embedding = None
            if self.share_nerf_encoding:
                if grid_coords is None:
                    raise ValueError('share_nerf_encoding requires grid_coords (decoder NeRF)')
                if dim != query_dim:
                    raise ValueError('share_nerf_encoding requires dim == query_dim')
            else:
                self.encoder_pos_encoder = NeRFPositionalEncoding(
                    encoder_grid_coords.shape[1], nerf_num_frequencies, output_dim=dim)
        else:
            self.encoder_grid_coords = None
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

        if grid_coords is not None:
            if grid_coords.shape[0] != n_out:
                raise ValueError('grid_coords first dim must equal n_out')
            self.register_buffer('grid_coords', grid_coords)
            self.query_encoder = NeRFPositionalEncoding(
                grid_coords.shape[1], nerf_num_frequencies, output_dim=query_dim)
            self.query_embedding = None
            effective_query_dim = self.query_encoder.output_dim
        else:
            self.grid_coords = None
            self.query_encoder = None
            self.query_embedding = nn.Parameter(torch.empty(1, n_out, query_dim))
            nn.init.trunc_normal_(self.query_embedding, std=0.02)
            effective_query_dim = query_dim

        if self.use_local_attention:
            DecoderBlock = LocalCrossAttentionDecoderBlock
        else:
            DecoderBlock = CrossAttentionDecoderBlock

        if decoder_cross_attn_mlp is None:
            use_cross_attn_mlp = (DecoderBlock is not CrossAttentionDecoderBlock)
        else:
            use_cross_attn_mlp = decoder_cross_attn_mlp

        decoder_blocks = []
        for i in range(decoder_depth):
            q_dim = effective_query_dim if i == 0 else dim
            decoder_blocks.append(DecoderBlock(
                q_dim, dim, num_heads, mlp_dim, dropout,
                use_cross_attn_mlp=use_cross_attn_mlp))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_norm = nn.LayerNorm(dim)

        self.output_head = nn.Linear(dim, num_vars)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Input batch of shape ``(B, C, H, W)`` matching ``x_shape``.

        Returns
        -------
        Tensor
            Predictions of shape ``(B, n_out, num_vars)``.
        """
        B = x.shape[0]

        # ENCODER

        tokens = self.patch_embedding(x)
        tokens = tokens.flatten(2).transpose(1, 2)    # (B, N_enc, dim)

        if self.encoder_grid_coords is not None:
            enc = self.query_encoder if self.share_nerf_encoding else self.encoder_pos_encoder
            tokens = tokens + enc(self.encoder_grid_coords).unsqueeze(0)
        else:
            tokens = tokens + self.pos_embedding
        tokens = self.dropout_emb(tokens)

        tokens = self.transformer_blocks(tokens)
        encoder_tokens = self.encoder_norm(tokens)     # (B, N_enc, dim)

        # DECODER

        if self.query_embedding is not None:
            queries = self.query_embedding.expand(B, -1, -1)       # (B, N_out, query_dim)
        else:
            queries = self.query_encoder(self.grid_coords)         # (N_out, query_dim)
            queries = queries.unsqueeze(0).expand(B, -1, -1)       # (B, N_out, query_dim)

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
