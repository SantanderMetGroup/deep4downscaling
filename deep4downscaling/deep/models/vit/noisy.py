# SPDX-License-Identifier: MIT

"""
This module contains the Noisy Vision Transformer (NoisyViT) model for statistical downscaling.

The model injects noise into the encoder to generate stochastic outputs. Two injection
mechanisms are supported: conditional layer normalization (Lang et al., 2024) and
multi-stage channel concatenation (NoisyDeepESD-style).

Lang, S., Alexe, M., Clare, M. C., Roberts, C., Adewoyin, R., Bouallègue, Z. B., ... & Leutbecher, M. (2024).
AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked
probability score. arXiv preprint arXiv:2412.15832.

Authors:
    Jose González-Abad
    Carlota García Fernández
""" 

import torch
import torch.nn as nn
import math

from .blocks import NoiseEmbedding, TransformerBlock, TransformerBlockCLN, CNNBlock, PixelShuffleDecoder

class NoisyViT(nn.Module):
    """
    Noisy Vision Transformer model for statistical downscaling. This model assumes that
    the spatial resolutions of the input and output tensors are powers of 2, and that
    the spatial resolution of the output is a multiple of the spatial resolution of the input.
    The model injects noise into the encoder to generate stochastic outputs following the
    implementation in Lang et al. (2024). The decoding from tokens to the high-resolution grid
    is selectable through the decoder argument.

    Lang, S., Alexe, M., Clare, M. C., Roberts, C., Adewoyin, R., Bouallègue, Z. B., ... & Leutbecher, M. (2024).
    AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked
    probability score. arXiv preprint arXiv:2412.15832.
    
    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must have dimension 4 (batch, channels, height, width).
        The spatial resolution must be a power of 2.

    y_shape : tuple
        Shape of the output data. Either 2D (batch, gridpoints) for univariate or
        3D (batch, num_vars, gridpoints) for multivariate. The spatial resolution must
        be both a power of 2 and a multiple of the spatial resolution of the input.

    patch_size : int
        Size of the patches to extract from the input image for building the token embeddings.
        The patch size must be a divisor of the spatial resolution of the input.

    dim : int
        Dimension of the token embeddings. This dimensions must be divisible by
        the number of heads.

    num_heads : int
        Number of attention heads within each transformer block.

    depth : int
        Number of transformer encoder blocks.

    mlp_dim : int
        Dimension of the MLP in transformer blocks.

    noise_channels : int
        Number of noise channels to inject into the input. Must be greater than 0.

    noise_dim : int
        Dimension of the noise embeddings. Only used when ``noise_injection='cln'``.

    members_for_training : int, optional
        Number of members to train in ensemble mode. Default is 2.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    orog : torch.Tensor, optional
        Orography data. Must have dimension 2 (height, width) and the same spatial resolution
        as the output data. If provided, the token decoding will be conditioned on the orography
        patches. When passed it must be already a torch.Tensor located in the same device as the model.

    decoder : str, optional
        Decoder used to map tokens to the high-resolution grid. Default is 'pixelshuffle'. Options:
        - 'pixelshuffle': PixelShuffle decoder followed by a convolutional tail operating at high
          resolution. The convolutions see across patch borders, removing the seams produced by
          independent per-token decoding.
        - 'linear': per-token linear decoder. Each token is decoded independently into a
          scale ** 2 patch and the patches are folded back into the high-resolution grid. This
          is the original decoder and can produce seams at the patch boundaries.
        Both decoders flatten the output in row-major (lat, lon) order, matching xarray's
        stack(gridpoint=('lat', 'lon')).

    noise_mode : str, optional
        Mode for noise injection. Default is 'patch'. Options:
        - 'patch': Different noise samples for each patch embedding (spatially varying).
        - 'global': Same noise sample for all patch embeddings (spatially uniform).

    noise_injection : str, optional
        Mechanism for noise injection. Default is 'cln'. Options:
        - 'cln': Conditional layer normalization in the transformer blocks, following
          Lang et al. (2024).
        - 'concat': Raw Gaussian noise channels concatenated to the input grid before
          patch embedding and to the decoder feature map before the pre-decoder CNN block.
          Uses plain transformer blocks without conditional normalization.

    num_vars : int, optional
        Number of output variables. Default is 1 (univariate, backward compatible).
        When > 1, the model outputs (B, num_vars, gridpoints). Can also be inferred
        from a 3D y_shape.

    last_relu : bool, optional
        If True, applies ReLU activation to the final output. Default is False.

    Notes
    -----
    The output grid is assumed to be square (H_out == W_out), so gridpoints must be
    a perfect square. When using the 'pixelshuffle' decoder, the upscaling factor
    (scale = H_out // H_tokens) must additionally be a power of 2.

    The forward pass returns a list of ensemble members when in training mode or
    when gradients are enabled (``self.training or torch.is_grad_enabled()``), and a
    single tensor otherwise. As a consequence, validation/evaluation must keep
    gradients enabled (do not wrap it in ``torch.no_grad()``) for the CRPS loss to
    receive several members; otherwise the loss collapses to a single member.
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim,  noise_channels, noise_dim,
                 members_for_training=2,
                 dropout=0., orog=None, decoder='pixelshuffle',
                 noise_mode='patch',
                 noise_injection='cln',
                 num_vars=1,
                 last_relu=False):
        super(NoisyViT, self).__init__()

        if len(x_shape) != 4:
            raise ValueError('X must be 4D (B, C, H, W)')

        if len(y_shape) == 2:
            gridpoints = y_shape[1]
        elif len(y_shape) == 3:
            num_vars = y_shape[1]
            gridpoints = y_shape[2]
        else:
            raise ValueError('Y must be 2D (B, gridpoints) or 3D (B, num_vars, gridpoints)')

        if x_shape[2] % patch_size != 0 or x_shape[3] % patch_size != 0:
            raise ValueError('Image dimensions must be divisible by patch_size')

        # Model parameters
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.members_for_training = members_for_training
        self.dropout = dropout
        self.orog = orog
        self.decoder_type = decoder
        self.num_vars = num_vars
        self.last_relu = last_relu

        # Noise injection parameters
        self.noise_channels = noise_channels
        self.noise_dim = noise_dim
        self.noise_mode = noise_mode
        self.noise_injection = noise_injection

        # Validate decoder
        if self.decoder_type not in ['pixelshuffle', 'linear']:
            raise ValueError("decoder must be either 'pixelshuffle' or 'linear'")

        # Validate noise_mode
        if self.noise_mode not in ['patch', 'global']:
            raise ValueError("noise_mode must be either 'patch' or 'global'")

        # Validate noise_injection
        if self.noise_injection not in ['cln', 'concat']:
            raise ValueError("noise_injection must be either 'cln' or 'concat'")

        # Coarse grid size (number of tokens in each dimension)
        self.H_tokens = x_shape[2] // patch_size
        self.W_tokens = x_shape[3] // patch_size
        self.num_patches = self.H_tokens * self.W_tokens

        # Target high-resolution size (square grid assumed)
        self.H_out = int(math.sqrt(gridpoints))
        self.W_out = self.H_out
        if self.H_out * self.W_out != gridpoints:
            raise ValueError("The output grid must be square: gridpoints must be a "
                             "perfect square (H_out == W_out)")

        # Upscaling factor
        self.scale = self.H_out // self.H_tokens

        if self.scale * self.H_tokens != self.H_out:
            raise ValueError("Output resolution must be divisible by input resolution")

        # Orography patch embedding
        if self.orog is not None:
            self.orography_embedding = nn.Linear(self.scale * self.scale, dim)

        # Patch embedding
        patch_in_channels = x_shape[1]
        if self.noise_injection == 'concat':
            patch_in_channels += noise_channels
        self.patch_embedding = nn.Conv2d(patch_in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Dropout for embeddings
        self.dropout_emb = nn.Dropout(dropout)

        # Noise embedding and transformer blocks
        if self.noise_injection == 'cln':
            self.noise_embedding = NoiseEmbedding(noise_channels, noise_dim)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlockCLN(dim, num_heads, mlp_dim, noise_dim, dropout)
                for _ in range(depth)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ])
            self.noise_proj = nn.Conv2d(dim + noise_channels, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

        # Pre-decoder CNN blocks
        self.cnn_block = CNNBlock(dim)

        # Decoder
        if self.decoder_type == 'pixelshuffle':
            self.decoder = PixelShuffleDecoder(dim, self.scale, out_channels=self.num_vars)
        else:
            self.kernel_size = self.scale

            # Per-token linear decoder (outputs num_vars * kernel_size**2 per token)
            self.token_decoder = nn.Linear(dim, self.num_vars * self.kernel_size**2)

            # Folding layer (folds num_vars * kernel_size**2 channels)
            self.fold = nn.Fold(output_size=(self.H_out, self.W_out),
                                kernel_size=self.kernel_size,
                                stride=self.scale)

    def forward(self, x):
        B = x.shape[0]

        # Determine if we are in ensemble mode
        is_ensemble_mode = self.training or torch.is_grad_enabled()

        # Set the number of members to iterate over
        if is_ensemble_mode:
            members_to_iterate = self.members_for_training
        else:
            members_to_iterate = 1

        out_members = []
        for i in range(members_to_iterate):

            if self.noise_injection == 'cln':
                # Sample noise
                if self.noise_mode == 'patch':
                    z = torch.randn(B, self.num_patches, self.noise_channels, device=x.device)
                else:
                    z = torch.randn(B, 1, self.noise_channels, device=x.device)
                    z = z.expand(-1, self.num_patches, -1)
                z = self.noise_embedding(z)

                # Patch embedding
                x_ = self.patch_embedding(x)
                x_ = x_.flatten(2).transpose(1, 2)

                # Add positional embeddings
                x_ = x_ + self.pos_embedding
                x_ = self.dropout_emb(x_)

                # Transformer
                for block in self.transformer_blocks:
                    x_ = block(x_, z)
                x_ = self.norm(x_)
            else:
                # Stage 1: concat noise to the input grid
                if self.noise_mode == 'patch':
                    z_in = torch.randn(B, self.noise_channels, x.shape[2], x.shape[3], device=x.device)
                else:
                    z_in = torch.randn(B, self.noise_channels, 1, 1, device=x.device)
                    z_in = z_in.expand(-1, -1, x.shape[2], x.shape[3])
                x_ = torch.cat((x, z_in), dim=1)

                x_ = self.patch_embedding(x_)
                x_ = x_.flatten(2).transpose(1, 2)

                # Add positional embeddings
                x_ = x_ + self.pos_embedding
                x_ = self.dropout_emb(x_)

                # Transformer
                for block in self.transformer_blocks:
                    x_ = block(x_)
                x_ = self.norm(x_)

            # Orography conditioning 
            if self.orog is not None:
                orog = self.orog.repeat(B, 1, 1)
                orog = orog.view(B, self.H_tokens, self.scale,
                                self.W_tokens, self.scale)
                orog = orog.permute(0, 1, 3, 2, 4).contiguous()
                orog = orog.view(B, self.H_tokens, self.W_tokens,
                                self.scale * self.scale)
                orog = orog.view(B, self.num_patches, self.scale * self.scale)
                orog_features = self.orography_embedding(orog)
                x_ = x_ + orog_features

            # Pre-decoder CNN block
            x_ = x_.transpose(1, 2).view(B, self.dim, self.H_tokens, self.W_tokens)

            if self.noise_injection == 'concat':
                if self.noise_mode == 'patch':
                    z_dec = torch.randn(B, self.noise_channels, self.H_tokens, self.W_tokens, device=x.device)
                else:
                    z_dec = torch.randn(B, self.noise_channels, 1, 1, device=x.device)
                    z_dec = z_dec.expand(-1, -1, self.H_tokens, self.W_tokens)
                x_ = torch.cat((x_, z_dec), dim=1)
                x_ = self.noise_proj(x_)

            x_ = self.cnn_block(x_)

            # Decoding to the high-resolution grid
            if self.decoder_type == 'pixelshuffle':
                # (B, num_vars, H_out, W_out)
                x_ = self.decoder(x_)
            else:
                # Per-token linear decoding: (B, num_patches, num_vars * kernel**2)
                x_ = x_.view(B, self.dim, self.num_patches).transpose(1, 2)
                x_ = self.token_decoder(x_)

                # Fold patches back into the high-resolution grid, per variable
                # (B, num_patches, num_vars * K**2) -> (B, num_vars * K**2, num_patches)
                x_ = x_.transpose(1, 2)
                x_ = x_.reshape(B * self.num_vars, self.kernel_size**2, self.num_patches)
                x_ = self.fold(x_)
                x_ = x_.view(B, self.num_vars, self.H_out, self.W_out)

            if self.last_relu:
                x_ = torch.relu(x_)

            if self.num_vars == 1:
                out = x_.view(B, -1)
            else:
                out = x_.view(B, self.num_vars, -1)

            out_members.append(out)

        if is_ensemble_mode:
            return out_members
        else:
            return out_members[0]