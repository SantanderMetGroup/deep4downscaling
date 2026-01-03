import torch
import torch.nn as nn
import math

from .blocks import NoiseEmbedding, TransformerBlockCLN, CNNBlock

class NoisyViT(nn.Module):
    """
    Noisy Vision Transformer model for statistical downscaling. This model assumes that
    the spatial resolutions of the input and output tensors are powers of 2, and that
    the spatial resolution of the output is a multiple of the spatial resolution of the input.
    The model injects noise into the encoder to generate stochastic outputs following the
    implementation in Lang et al. (2024).

    Lang, S., Alexe, M., Clare, M. C., Roberts, C., Adewoyin, R., Bouallègue, Z. B., ... & Leutbecher, M. (2024).
    AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked
    probability score. arXiv preprint arXiv:2412.15832.
    
    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must have dimension 4 (batch, channels, height, width).
        The spatial resolution must be a power of 2.

    y_shape : tuple
        Shape of the output data. Must have dimension 2 (batch, num_outputs).
        The spatial resolution must both a power of 2 and a multiple of the
        spatial resolution of the input.

    patch_size : int
        Size of the patches to extract from the input image.
        The patch size must be a divisor of the spatial resolution of the input.

    dim : int
        Dimension of the token embeddings. This dimensions must be divisible by
        the number of heads.

    depth : int
        Number of transformer encoder blocks.

    num_heads : int
        Number of attention heads.

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
        patches.

    overlap : int, optional
        Overlap between patches. Default is 0. This is used to create a smooth transition
        between patches, thus avoiding artifacts at the boundaries of the patches.

    last_relu : bool, optional
        If set to True, the output of the last linear decoder is passed through a
        ReLU activation function. By default is set to False.

    Notes
    -----
    The model uses a per-token linear decoder at the end that transforms each
    token embedding into a spatial patch of size (scale * scale + 2 * overlap), which are
    then reshaped and tiled to form the final high-resolution output.
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim,  noise_channels, noise_dim,
                 members_for_training=2,
                 dropout=0., orog=None, overlap=0,
                 last_relu=False):
        super(NoisyViT, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            raise ValueError('X must be 4D (B, C, H, W) and Y must be 2D (B, N_outputs)')

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
        self.overlap = overlap
        self.last_relu = last_relu

        # CLN parameters
        self.noise_channels = noise_channels
        self.noise_dim = noise_dim

        # Coarse grid size (number of tokens in each dimension)
        self.H_tokens = x_shape[2] // patch_size
        self.W_tokens = x_shape[3] // patch_size
        self.num_patches = self.H_tokens * self.W_tokens

        # Target high-resolution size
        self.H_out = int(math.sqrt(y_shape[1]))
        self.W_out = self.H_out

        # Upscaling factor
        self.scale = self.H_out // self.H_tokens

        if self.scale * self.H_tokens != self.H_out:
            raise ValueError("Output resolution must be divisible by input resolution")

        # Overlapping patch parameters
        self.kernel_size = self.scale + 2 * self.overlap

        # Orography patch embedding: projects (scale * scale) patch to token dimension
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

        # Per-token linear decoder: outputs patches (possibly overlapping)
        self.token_decoder = nn.Linear(dim, self.scale**2)

        # Folding layer for overlap-add reconstruction
        self.fold = nn.Fold(output_size=(self.H_out, self.W_out),
                            kernel_size=self.kernel_size,
                            padding=self.overlap,
                            stride=self.scale)

        # Windowing and normalization mask
        if self.overlap > 0:
            window = torch.hann_window(self.kernel_size, periodic=False)
            window = window.unsqueeze(0) * window.unsqueeze(1)
            self.register_buffer('window', window.view(-1, 1))
        else:
            self.register_buffer('window', torch.ones(self.scale**2, 1))

        # Pre-compute normalization mask to handle overlapping regions
        ones = torch.ones(1, 1, self.num_patches)
        self.register_buffer('norm_mask', self.fold(self.window * ones))

    def forward(self, x, orography=None):
        
        B = x.shape[0]

        is_ensemble_mode = self.training or torch.is_grad_enabled()

        if is_ensemble_mode:
            members_to_iterate = self.members_for_training
        else:
            members_to_iterate = 1

        out_members = []
        for i in range(members_to_iterate):

            # Sample noise
            z = torch.randn(B, self.num_patches, self.noise_channels, device=x.device)
            z = self.noise_embedding(z)

            # Patch embedding
            x_ = self.patch_embedding(x)                 # (B, D, Ht, Wt)
            x_ = x_.flatten(2).transpose(1, 2)            # (B, N, D)

            # Add positional embeddings
            x_ = x_ + self.pos_embedding                  # (B, N, D)
            x_ = self.dropout_emb(x_)                     # (B, N, D)

            # Transformer
            for block in self.transformer_blocks:
                x_ = block(x_, z)
            x_ = self.norm(x_)                            # (B, N, D)

            # Orography conditioning (if provided)
            if self.orog is not None:
                # Replicate across batch dimension
                orog = self.orog.repeat(B, 1, 1)

                # (B, H_out, W_out) -> (B, H_tokens, scale, W_tokens, scale)
                orog = orog.view(B, self.H_tokens, self.scale,
                                self.W_tokens, self.scale)

                # Permute to group patches: (B, H_tokens, W_tokens, scale, scale)
                orog = orog.permute(0, 1, 3, 2, 4).contiguous()

                # Flatten each patch: (B, H_tokens, W_tokens, scale * scale)
                orog = orog.view(B, self.H_tokens, self.W_tokens,
                                self.scale * self.scale)

                # Flatten spatial token dimensions: (B, num_patches, scale * scale)
                orog = orog.view(B, self.num_patches, self.scale * self.scale)
                
                # Project orography patches to token dimension
                orog_features = self.orography_embedding(orog)  # (B, N, D)
                
                # Add orography features to token embeddings
                x_ = x_ + orog_features

            # Pre-decoder CNN block
            x_ = x_.transpose(1, 2).view(B, self.dim, self.H_tokens, self.W_tokens)     
            x_ = self.cnn_block(x_)
            x_ = x_.view(B, self.dim, self.num_patches).transpose(1, 2)

            # Per-token decoding
            x_ = self.token_decoder(x_)                   # (B, N, kernel_size**2)

            # Overlap-add reconstruction
            # TODO: For some reason this is required for CRPS_SPECTRAL loss to work.
            x_ = x_.transpose(1, 2)                       # (B, kernel_size**2, N)
            x_ = x_ * self.window                         # Apply window
            x_ = self.fold(x_)                          # Fold into spatial grid
            x_ = x_ / self.norm_mask.clamp(min=1e-8)  # Normalize blended regions

            if self.last_relu:
                x_ = torch.relu(x_)

            out = x_.view(B, -1)

            out_members.append(out)

        if is_ensemble_mode:
            return out_members
        else:
            return out_members[0]