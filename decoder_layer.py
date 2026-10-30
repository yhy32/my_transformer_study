import torch
import torch.nn as nn

from feed_forward import PositionWiseFeedForward
from multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )

        self.cross_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(
            x, x, x, mask=tgt_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))

        cross_output, _= self.cross_attn(
            x, enc_output, enc_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout2(cross_output))

        ffn_output = self.feed_forward(
            x
        )
        x = self.norm3(x + self.dropout3(ffn_output))

        return x
