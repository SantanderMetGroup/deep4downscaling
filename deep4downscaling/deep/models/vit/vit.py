import torch
import torch.nn as nn
import math

from .blocks import TransformerBlock


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for statistical downscaling. This model assumes that
    the spatial resolutions of the input and output tensors are powers of 2, and that
    the spatial resolution of the output is a multiple of the spatial resolution of the input.
    
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

    num_heads : int
        Number of attention heads.

    depth : int
        Number of transformer encoder blocks.

    mlp_dim : int
        Dimension of the MLP in transformer blocks.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    orog : torch.Tensor, optional
        Orography data. Must have dimension 2 (height, width) and the same spatial resolution
        as the output data. If provided, the token decoding will be conditioned on the orography
        patches.

    overlap : int, optional
        Overlap between patches. Default is 0. This is used to create a smooth transition
        between patches, thus avoiding artifacts at the boundaries of the patches.

    Notes
    -----
    The model uses a per-token linear decoder at the end that transforms each
    token embedding into a spatial patch of size (scale * scale + 2 * overlap), which are
    then reshaped and tiled to form the final high-resolution output.
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim, dropout=0., orog=None, overlap=0):
        super(ViT, self).__init__()

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
        self.dropout = dropout
        self.orog = orog
        self.overlap = overlap

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

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Per-token linear decoder: outputs patches (possibly overlapping)
        self.token_decoder = nn.Linear(dim, self.kernel_size**2)

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

        # Patch embedding
        x = self.patch_embedding(x)                 # (B, D, Ht, Wt)
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)

        # Add positional embeddings
        x = x + self.pos_embedding                  # (B, N, D)
        x = self.dropout_emb(x)                     # (B, N, D)

        # Transformer
        x = self.transformer_blocks(x)              # (B, N, D)
        x = self.norm(x)                            # (B, N, D)

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
            x = x + orog_features

        # Per-token decoding
        x = self.token_decoder(x)                   # (B, N, kernel_size**2)

        # Overlap-add reconstruction
        x = x.transpose(1, 2)                       # (B, kernel_size**2, N)
        x = x * self.window                         # Apply window
        out = self.fold(x)                          # Fold into spatial grid
        out = out / self.norm_mask.clamp(min=1e-8)  # Normalize blended regions

        return out.view(B, -1)

