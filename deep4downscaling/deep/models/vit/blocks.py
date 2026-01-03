import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
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

    def __init__(self, dim, noise_dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.gamma = nn.Linear(noise_dim, dim)
        self.beta = nn.Linear(noise_dim, dim)

    def forward(self, x, z):
        x_norm = self.norm(x)
        gamma = self.gamma(z)
        beta = self.beta(z)
        return gamma * x_norm + beta

class TransformerBlockCLN(nn.Module):
    """Transformer encoder block with multi-head attention and MLP, conditioned on noise
       through conditional layer normalization."""

    def __init__(self, dim, num_heads, mlp_dim, noise_dim, dropout=0.):
        super().__init__()

        self.norm1 = ConditionalLayerNorm(dim, noise_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)

        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x, z):
        x = x + self.attn(self.norm1(x, z))
        x = x + self.mlp(x)
        return x

class CNNBlock(nn.Module):
    """CNN Block. Conv2d, GELU, Conv2d."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        return x + self.block(x)