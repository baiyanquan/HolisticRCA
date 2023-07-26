import torch
import torch.nn as nn
import math


class PrepareForSelfAttention(nn.Module):
    def __init__(self, o11y_length: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(o11y_length, d_k, bias=bias)
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model
        self.query = PrepareForSelfAttention(d_model, self.d_k, bias=bias)
        self.key = PrepareForSelfAttention(d_model, self.d_k, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibd,jbd->ijb', query, key)

    def forward(self, x: torch.Tensor):
        query = self.query(x)
        key = self.key(x)
        scores = self.get_scores(query, key)
        scores *= self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum("ijb,jbd->ibd", attn, x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, o11y_length: int, window_size: int, d_model: int, dropout_prob: float):
        self.self_attn = SelfAttention(d_model=d_model)

