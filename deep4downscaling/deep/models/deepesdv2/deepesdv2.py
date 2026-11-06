# SPDX-License-Identifier: MIT

"""
DeepESDv2: a Vision Transformer encoder with a cross-attention decoder for
statistical downscaling and RCM emulation.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn

from ..vit.blocks import TransformerBlock
from .blocks import ConvStem, CrossAttentionDecoderBlock, NeRFPositionalEncoding


class DeepESDv2(nn.Module):
    """
    DeepESDv2 maps a coarse predictor field to a high-resolution target by
    querying every output point with its location.

    Architecture::

        predictors (B, C, H, W)
          └─ ConvStem (two 3x3 convs)            → one token per predictor grid point
          └─ + NeRF(lat, lon of the grid point)  → (B, H*W, dim)
          └─ ViT encoder (depth x TransformerBlock) + LayerNorm → tokens

        query_i = [ NeRF(lat_i, lon_i) ‖ emb(covariable_1,i) ‖ ... ]   (dim)
          └─ decoder_depth x [ cross-attention to tokens + MLP ]   (residual, pre-norm)
          └─ Linear head (+ ReLU)                → (B, n_out[, num_vars])

    Design choices:

    * **Location queries.** Each output point is a query built from a NeRF
      Fourier encoding of its coordinates (Mildenhall et al., 2020) plus
      embeddings of static covariables (e.g. orography). The output grid is
      therefore defined only by ``grid_coords``: points are decoded
      independently (queries do not interact), so any subset can be decoded
      with ``query_idx`` (Perceiver IO, Jaegle et al., 2022).
    * **Convolutional stem.** Tokens are built with two 3x3 convolutions, so
      each token sees its neighbourhood of the predictor field.
    * **Unbounded head.** There is no LayerNorm before the linear head: since
      a LayerNorm output has fixed norm, ``LayerNorm -> Linear`` bounds the
      prediction (``|y| <= sqrt(dim) * ||w * gamma|| + const``), which caps
      extremes of heavy-tailed variables such as precipitation.
    * **Identity start.** The cross-attention output projections are
      zero-initialised, so at initialisation the decoder passes the queries
      through unchanged and the model starts from a static (climatology-like)
      field.

    Coordinates (``grid_coords``, ``encoder_grid_coords``) should share one
    reference frame and be scaled to roughly ``[-1, 1]``, for example lat/lon
    mapped with a common bounding box.

    Parameters
    ----------
    x_shape : tuple
        Shape of the input, (batch, channels, height, width).

    n_out : int
        Number of output grid points.

    grid_coords : torch.Tensor
        (n_out, coord_dim) coordinates of the output grid points, in the order
        of the model output.

    encoder_grid_coords : torch.Tensor
        (height * width, coord_dim) coordinates of the predictor grid points,
        in row-major (height, width) order.

    dim : int, optional
        Token and query dimension. Default 256.

    depth : int, optional
        Number of encoder transformer blocks. Default 4.

    num_heads : int, optional
        Attention heads in the encoder and the decoder. Default 4.

    mlp_dim : int, optional
        Hidden size of the encoder and decoder MLPs. Default 1024.

    decoder_depth : int, optional
        Number of cross-attention decoder blocks. Default 2.

    nerf_num_frequencies : int, optional
        Frequency octaves of the NeRF encodings. Default 8.

    covariables : torch.Tensor or None, optional
        (n_out, n_covars) static fields at the output points (e.g. orography),
        scaled to roughly ``[-1, 1]``. Each column is embedded by its own
        ``Linear(1, covar_embed_dim)`` and concatenated to the query. Default
        None.

    covar_embed_dim : int, optional
        Embedding size per covariable. The NeRF part of the query gets
        ``dim - n_covars * covar_embed_dim`` dimensions. Default 16.

    num_vars : int, optional
        Number of output variables. Default 1.

    dropout : float, optional
        Dropout probability. Default 0.

    last_relu : bool, optional
        Apply a ReLU to the output (e.g. for precipitation). Default False.
    """

    def __init__(self, x_shape, n_out, grid_coords, encoder_grid_coords,
                 dim=256, depth=4, num_heads=4, mlp_dim=1024, decoder_depth=2,
                 nerf_num_frequencies=8, covariables=None, covar_embed_dim=16,
                 num_vars=1, dropout=0., last_relu=False):
        super().__init__()

        if len(x_shape) != 4:
            raise ValueError('x_shape must be 4D (B, C, H, W)')
        n_tokens = x_shape[2] * x_shape[3]
        if encoder_grid_coords.shape[0] != n_tokens:
            raise ValueError(f'encoder_grid_coords must have {n_tokens} rows (H * W)')
        if grid_coords.shape[0] != n_out:
            raise ValueError('grid_coords must have n_out rows')
        if covariables is not None and covariables.shape[0] != n_out:
            raise ValueError('covariables must have n_out rows')

        n_covars = 0 if covariables is None else covariables.shape[1]
        query_pos_dim = dim - n_covars * covar_embed_dim
        if query_pos_dim <= 0:
            raise ValueError('dim must exceed n_covars * covar_embed_dim')

        self.num_vars = num_vars
        self.last_relu = last_relu

        # ENCODER
        self.stem = ConvStem(x_shape[1], dim)
        self.register_buffer('encoder_grid_coords', encoder_grid_coords)
        self.encoder_pos = NeRFPositionalEncoding(encoder_grid_coords.shape[1],
                                                  nerf_num_frequencies, output_dim=dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(*[TransformerBlock(dim, num_heads, mlp_dim, dropout)
                                       for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(dim)

        # QUERIES
        self.register_buffer('grid_coords', grid_coords)
        self.query_pos = NeRFPositionalEncoding(grid_coords.shape[1], nerf_num_frequencies,
                                                output_dim=query_pos_dim)
        if covariables is not None:
            self.register_buffer('covariables', covariables)
            self.covar_embeddings = nn.ModuleList([nn.Linear(1, covar_embed_dim)
                                                   for _ in range(n_covars)])
        else:
            self.covariables = None

        # DECODER
        self.decoder = nn.ModuleList([CrossAttentionDecoderBlock(dim, num_heads, mlp_dim, dropout)
                                      for _ in range(decoder_depth)])
        self.head = nn.Linear(dim, num_vars)

    def encode(self, x):
        """Predictors (B, C, H, W) -> encoder tokens (B, H*W, dim)."""
        tokens = self.stem(x).flatten(2).transpose(1, 2)                    # (B, H*W, dim)
        tokens = tokens + self.encoder_pos(self.encoder_grid_coords).unsqueeze(0)
        tokens = self.embed_dropout(tokens)
        return self.encoder_norm(self.encoder(tokens))

    def queries(self, query_idx=None):
        """Decoder queries (N_q, dim) for all output points or the subset ``query_idx``."""
        coords = self.grid_coords if query_idx is None else self.grid_coords[query_idx]
        queries = self.query_pos(coords)
        if self.covariables is not None:
            cov = self.covariables if query_idx is None else self.covariables[query_idx]
            embs = [fc(cov[:, i:i + 1]) for i, fc in enumerate(self.covar_embeddings)]
            queries = torch.cat([queries, *embs], dim=-1)
        return queries

    def forward(self, x, query_idx=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Predictors of shape (B, C, H, W).

        query_idx : torch.LongTensor or None, optional
            Output points to decode. Points are decoded independently, so this
            equals decoding all points and selecting ``query_idx``; useful to
            train on random subsets of the grid. Default None (all points).

        Returns
        -------
        torch.Tensor
            (B, N_q, num_vars), or (B, N_q) when ``num_vars == 1``.
        """
        tokens = self.encode(x)
        queries = self.queries(query_idx).unsqueeze(0).expand(x.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, tokens)
        out = self.head(queries)

        if self.last_relu:
            out = torch.relu(out)
        if self.num_vars == 1:
            out = out.squeeze(-1)
        return out
