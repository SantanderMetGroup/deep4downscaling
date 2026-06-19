# SPDX-License-Identifier: MIT

"""
This module contains the Noisy Vision Transformer (NoisyViT) model for statistical downscaling.

The model injects noise into the encoder to generate stochastic outputs following the
implementation in Lang et al. (2024).

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

from .blocks import NoiseEmbedding, TransformerBlockCLN, CNNBlock, PixelShuffleDecoder

class NoisyViT(nn.Module):
    """
    Noisy Vision Transformer model for statistical downscaling. This model assumes that
    the spatial resolutions of the input and output tensors are powers of 2, and that
    the spatial resolution of the output is a multiple of the spatial resolution of the input.
    The model injects noise into the encoder to generate stochastic outputs following the
    implementation in Lang et al. (2024). The decoding from tokens to the high-resolution grid
    is selectable through the decoder argument (see below).

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
        Dimension of the noise embeddings.

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
          independent per-token decoding. The output is flattened in row-major (lat, lon) order,
          matching xarray's stack(gridpoint=('lat', 'lon')). The overlap argument is ignored.
        - 'linear': per-token linear decoder with overlap-add reconstruction. Each token is decoded
          independently into a (scale + 2 * overlap) ** 2 patch, and the patches are folded back
          with overlap-add. The output is flattened in (token, intra-patch) order. This is the
          original decoder and can produce seams at the patch boundaries.

    overlap : int, optional
        Overlap between patches. Default is 0. Only used when decoder is 'linear'. This is used to
        create a smooth transition between patches, thus avoiding artifacts at the boundaries of the
        patches. This issue is especially noticeable when injecting noise, as this noise is injected
        independently in each patch embedding. (See Notes for more details.)

    noise_mode : str, optional
        Mode for noise injection. Default is 'patch'. Options:
        - 'patch': Different noise samples for each patch embedding (spatially varying).
        - 'global': Same noise sample for all patch embeddings (spatially uniform).

    num_vars : int, optional
        Number of output variables. Default is 1 (univariate, backward compatible).
        When > 1, the model outputs (B, num_vars, gridpoints). Can also be inferred
        from a 3D y_shape.

    last_relu : bool, optional
        If True, applies ReLU activation to the final output. Default is False.

    Notes
    -----
    Overlap-Add Reconstruction (only applicable when decoder is 'linear' and overlap > 0):
    1. Each token decodes to enlarged (scale + 2 * overlap)**2 patches
    2. Hann window applied: strong at center, fades to zero at edges
    3. Patches placed with stride=scale, overlapping regions are summed
    4. Normalization divides by accumulated weights to get proper average
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim,  noise_channels, noise_dim,
                 members_for_training=2,
                 dropout=0., orog=None, decoder='pixelshuffle', overlap=0,
                 noise_mode='patch',
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
        self.overlap = overlap
        self.num_vars = num_vars
        self.last_relu = last_relu

        # Noise injection parameters
        self.noise_channels = noise_channels
        self.noise_dim = noise_dim
        self.noise_mode = noise_mode

        # Validate decoder
        if self.decoder_type not in ['pixelshuffle', 'linear']:
            raise ValueError("decoder must be either 'pixelshuffle' or 'linear'")

        # Validate noise_mode
        if self.noise_mode not in ['patch', 'global']:
            raise ValueError("noise_mode must be either 'patch' or 'global'")

        # Coarse grid size (number of tokens in each dimension)
        self.H_tokens = x_shape[2] // patch_size
        self.W_tokens = x_shape[3] // patch_size
        self.num_patches = self.H_tokens * self.W_tokens

        # Target high-resolution size
        self.H_out = int(math.sqrt(gridpoints))
        self.W_out = self.H_out

        # Upscaling factor
        self.scale = self.H_out // self.H_tokens

        if self.scale * self.H_tokens != self.H_out:
            raise ValueError("Output resolution must be divisible by input resolution")

        # Orography patch embedding
        if self.orog is not None:
            self.orography_embedding = nn.Linear(self.scale * self.scale, dim)

        # Patch embedding
        self.patch_embedding = nn.Conv2d(x_shape[1], dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Dropout for embeddings
        self.dropout_emb = nn.Dropout(dropout)

        # Noise embedding
        self.noise_embedding = NoiseEmbedding(noise_channels, noise_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockCLN(dim, num_heads, mlp_dim, noise_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Pre-decoder CNN blocks
        self.cnn_block = CNNBlock(dim)

        # Decoder
        if self.decoder_type == 'pixelshuffle':
            self.decoder = PixelShuffleDecoder(dim, self.scale, out_channels=self.num_vars)
        else:
            # Overlap-add reconstruction parameters
            self.kernel_size = self.scale + 2 * self.overlap

            # Per-token linear decoder (outputs num_vars * kernel_size**2 per token)
            self.token_decoder = nn.Linear(dim, self.num_vars * self.kernel_size**2)

            # Folding layer (folds num_vars * kernel_size**2 channels)
            self.fold = nn.Fold(output_size=(self.H_out, self.W_out),
                                kernel_size=self.kernel_size,
                                padding=self.overlap,
                                stride=self.scale)

            # Windowing: tile the 2D window across num_vars channels
            if self.overlap > 0:
                window = torch.hann_window(self.kernel_size, periodic=False)
                window = window.unsqueeze(0) * window.unsqueeze(1)
                window_1var = window.view(-1, 1)
            else:
                window_1var = torch.ones(self.kernel_size**2, 1)
            self.register_buffer('window', window_1var.repeat(self.num_vars, 1))

            # Pre-compute normalization mask (broadcasts over the variable channel)
            ones = torch.ones(1, self.kernel_size**2, self.num_patches)
            self.register_buffer('norm_mask', self.fold(window_1var * ones))

    def forward(self, x, orography=None):
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
            x_ = self.cnn_block(x_)

            # Decoding to the high-resolution grid
            if self.decoder_type == 'pixelshuffle':
                # (B, num_vars, H_out, W_out)
                x_ = self.decoder(x_)
            else:
                # Per-token linear decoding: (B, num_patches, num_vars * kernel**2)
                x_ = x_.view(B, self.dim, self.num_patches).transpose(1, 2)
                x_ = self.token_decoder(x_)

                # Overlap-add reconstruction per variable
                # (B, num_patches, num_vars * K**2) -> (B, num_vars * K**2, num_patches)
                x_ = x_.transpose(1, 2)
                x_ = x_ * self.window
                # Fold each variable independently
                x_ = x_.reshape(B * self.num_vars, self.kernel_size**2, self.num_patches)
                x_ = self.fold(x_)
                x_ = x_ / self.norm_mask.clamp(min=1e-8)
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