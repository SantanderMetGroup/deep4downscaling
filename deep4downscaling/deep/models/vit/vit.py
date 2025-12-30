import torch
import torch.nn as nn
import math

from .blocks import TransformerBlock


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for statistical downscaling.

    Parameters
    ----------
    x_shape : tuple
        Shape of the input data. Must have dimension 4 (batch, channels, height, width).

    y_shape : tuple
        Shape of the output data. Must have dimension 2 (batch, num_outputs).

    patch_size : int
        Size of the patches to extract from the input image.

    dim : int
        Dimension of the token embeddings.

    depth : int
        Number of transformer encoder blocks.

    num_heads : int
        Number of attention heads.

    mlp_dim : int
        Dimension of the MLP in transformer blocks.

    dropout : float, optional
        Dropout probability. Default is 0.0.

    Notes
    -----
    The model uses a per-token linear decoder at the end that transforms each
    token embedding into a spatial patch of size (scale * scale), which are
    then reshaped and tiled to form the final high-resolution output.
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim, dropout=0.):
        super(ViT, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            raise ValueError(
                'X must be 4D (B, C, H, W) and Y must be 2D (B, N_outputs)'
            )

        if x_shape[2] % patch_size != 0 or x_shape[3] % patch_size != 0:
            raise ValueError('Image dimensions must be divisible by patch_size')

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Coarse grid size (number of tokens in each dimension)
        self.H_tokens = x_shape[2] // patch_size
        self.W_tokens = x_shape[3] // patch_size
        self.num_patches = self.H_tokens * self.W_tokens

        # Target high-resolution size
        self.H_out = int(math.sqrt(y_shape[1]))
        self.W_out = self.H_out

        # Upscaling factor (assumed integer)
        self.scale = self.H_out // self.H_tokens

        if self.scale * self.H_tokens != self.H_out:
            raise ValueError("Output resolution must be divisible by input resolution")

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

        # Per-token linear decoder
        self.token_decoder = nn.Linear(dim, self.scale * self.scale)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)                 # (B, D, Ht, Wt)
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)

        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout_emb(x)

        # Transformer
        x = self.transformer_blocks(x)
        x = self.norm(x)                            # (B, N, D)

        # Per-token decoding
        x = self.token_decoder(x)                   # (B, N, r**2 * C)

        # Reshape tokens → spatial field
        # (B, N, r**2) → (B, Ht, Wt, r, r)
        x = x.view(B, self.H_tokens, self.W_tokens,
                   self.scale, self.scale)

        # Tile patches
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        out = x.view(B, self.H_out * self.W_out)

        return out
