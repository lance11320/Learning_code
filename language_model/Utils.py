import torch
import torch.nn as nn
import math
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-1, -2))

        attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, 1e-9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        return context, attention

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,embed_dim = 512,ffdim = 1024):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ffdim, bias=False),
            nn.ReLU(),
            nn.Linear(ffdim, embed_dim, bias=False)
        )
        self.layernorm=nn.LayerNorm(embed_dim)
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual) # [batch_size, seq_len, d_model]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

