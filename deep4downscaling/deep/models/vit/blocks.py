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