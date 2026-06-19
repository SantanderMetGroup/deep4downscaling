# SPDX-License-Identifier: MIT

"""
This module contains the building blocks for the Vision Transformer (ViT) and Noisy Vision Transformer (NoisyViT) models.

Authors:
    Jose González-Abad
    Carlota García Fernández
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(q, k, v,
                                                     dropout_p=self.attn_dropout if self.training else 0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, dim)
        output = self.out_proj(attn_output)

        return output

class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention and MLP."""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()

        self.attention = nn.Sequential(nn.LayerNorm(dim),
                                       MultiHeadAttention(dim, num_heads, dropout))

        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

class NoiseEmbedding(nn.Module):
    """Noise Embedding."""

    def __init__(self, noise_channels, noise_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(noise_channels, noise_dim),
                                 nn.GELU(),
                                 nn.Linear(noise_dim, noise_dim))
        self.norm = nn.LayerNorm(noise_dim)

    def forward(self, xi):
        z = self.mlp(xi)
        return self.norm(z)

class ConditionalLayerNorm(nn.Module):
    """Conditional Layer Normalization."""

    def __init__(self, dim, noise_dim, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.gamma = nn.Linear(noise_dim, dim)
        self.beta = nn.Linear(noise_dim, dim)
        
        if zero_init:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    def forward(self, x, z):
        x_norm = self.norm(x)
        gamma = self.gamma(z)
        beta = self.beta(z)
        return (1 + gamma) * x_norm + beta

class TransformerBlockCLN(nn.Module):
    """Transformer encoder block with multi-head attention and MLP, conditioned on noise
       through conditional layer normalization."""

    def __init__(self, dim, num_heads, mlp_dim, noise_dim, dropout=0.):
        super().__init__()

        self.norm1 = ConditionalLayerNorm(dim, noise_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm2 = ConditionalLayerNorm(dim, noise_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x, z):
        x = x + self.attn(self.norm1(x, z))
        x = x + self.mlp(self.norm2(x, z))
        return x

class CNNBlock(nn.Module):
    """Standard CNN Block. Conv2d, GELU, Conv2d."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        return x + self.block(x)

class PixelShuffleDecoder(nn.Module):
    """PixelShuffle decoder with a convolutional tail at high resolution (ESPCN/EDSR-style).

       The token grid (B, dim, H_tokens, W_tokens) is progressively upsampled by factors
       of 2 (Conv2d, PixelShuffle, GELU) up to the high-resolution grid, and then refined
       with plain 3x3 convolutions operating at full resolution. The number of channels is
       halved at each upsampling stage (with a floor of 32) to keep the computation at high
       resolution tractable. The convolutions preceding each PixelShuffle are initialized with
       ICNR (Aitken et al., 2017) to suppress checkerboard artifacts."""

    def __init__(self, dim, scale, out_channels=1):
        super().__init__()

        if scale < 1 or (scale & (scale - 1)) != 0:
            raise ValueError('scale must be a power of 2')

        # Progressive x2 upsampling stages
        upsampling = []
        channels = dim
        for _ in range(int(math.log2(scale))):
            stage_out = max(channels // 2, 32)
            conv = nn.Conv2d(channels, stage_out * 4, kernel_size=3, padding=1)
            self._icnr_init(conv.weight, upscale_factor=2)
            upsampling.extend([conv, nn.PixelShuffle(2), nn.GELU()])
            channels = stage_out
        self.upsampling = nn.Sequential(*upsampling)

        # Convolutional tail at high resolution
        self.tail = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))

    @staticmethod
    def _icnr_init(weight, upscale_factor):
        """ICNR initialization (Aitken et al., 2017). Makes the Conv2d + PixelShuffle pair
           equivalent to nearest-neighbour upsampling at initialization."""
        out_channels, in_channels, h, w = weight.shape
        sub_kernel = torch.empty(out_channels // upscale_factor**2, in_channels, h, w)
        nn.init.kaiming_normal_(sub_kernel)
        with torch.no_grad():
            weight.copy_(sub_kernel.repeat_interleave(upscale_factor**2, dim=0))

    def forward(self, x):
        x = self.upsampling(x)
        return self.tail(x)