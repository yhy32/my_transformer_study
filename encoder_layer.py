import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=0.0
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attn(
            x, x, x, mask=src_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x