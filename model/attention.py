import torch
import torch.nn as nn

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class VectorAttention(nn.Module):
    """
    一维向量注意力机制
    """
    def __init__(self, input_dim: int, attn_drop_rate: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.qkv_bias = qkv_bias

        self.linear_q = nn.Sequential(
                    nn.Linear(input_dim, input_dim, bias=True),
                    nn.BatchNorm1d(input_dim),
                    nn.ReLU(inplace=True)
                )
        self.linear_k = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_v = nn.Linear(input_dim, input_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.input_dim ** 0.5)  # [B, N, N]
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, value)
        return out + feat

class MultiHeadVectorAttention(nn.Module):
    """
    One-dimensional multi-head attention mechanism. Ref: https://arxiv.org/abs/1706.03762
    """