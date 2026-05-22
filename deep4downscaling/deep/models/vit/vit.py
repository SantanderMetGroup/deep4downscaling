# SPDX-License-Identifier: MIT

"""
This module contains the Vision Transformer (ViT) model for statistical downscaling.

Authors:
    Jose González-Abad
    Carlota García Fernández
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .blocks import TransformerBlock, CNNBlock

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

    dropout : float, optional
        Dropout probability. Default is 0.0.

    orog : torch.Tensor, optional
        Orography data. Must have dimension 2 (height, width) and the same spatial resolution
        as the output data. If provided, the token decoding will be conditioned on the orography
        patches. When passed it must be already a torch.Tensor located in the same device as the model.

    decoder_neighborhood : int, optional
        Radius of the spatial neighborhood used for per-token decoding. Default is 0,
        which decodes each token independently. When greater than 0, for each token the
        decoder receives the concatenation of all tokens within a
        (2 * decoder_neighborhood + 1) x (2 * decoder_neighborhood + 1) spatial window
        centered on that token. Border tokens are zero-padded.

    last_relu : bool, optional
        If True, applies ReLU activation to the final output. Default is False.
    """

    def __init__(self, x_shape, y_shape, patch_size, dim, depth, num_heads,
                 mlp_dim, dropout=0., orog=None, decoder_neighborhood=0,
                 last_relu=False):
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
        self.decoder_neighborhood = decoder_neighborhood
        self.last_relu = last_relu
        
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

        # Pre-decoder CNN blocks
        self.cnn_block = CNNBlock(dim)

        # Per-token linear decoder
        decoder_input_dim = dim * (2 * decoder_neighborhood + 1)**2 if decoder_neighborhood > 0 else dim
        self.token_decoder = nn.Linear(decoder_input_dim, self.scale**2)

    def forward(self, x, orography=None):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)                 
        x = x.flatten(2).transpose(1, 2)            

        # Add positional embeddings
        x = x + self.pos_embedding                  
        x = self.dropout_emb(x)                     

        # Transformer
        x = self.transformer_blocks(x)              
        x = self.norm(x)                            

        # Orography conditioning 
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
            orog_features = self.orography_embedding(orog) 
            
            # Add orography features to token embeddings
            x = x + orog_features

        # Pre-decoder CNN block
        x = x.transpose(1, 2).view(B, self.dim, self.H_tokens, self.W_tokens)     
        x = self.cnn_block(x)

        # Gather token neighborhoods (if applicable) and decode
        if self.decoder_neighborhood > 0:
            k = 2 * self.decoder_neighborhood + 1
            x = F.unfold(x, kernel_size=k, padding=self.decoder_neighborhood)
            x = x.transpose(1, 2)
        else:
            x = x.view(B, self.dim, self.num_patches).transpose(1, 2)

        x = self.token_decoder(x)               
            
        if self.last_relu:
            x = torch.relu(x)
        
        return x.view(B, -1)