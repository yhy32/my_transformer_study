import math
import torch 
import torch.nn as nn

from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from masking import create_padding_mask

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, src):
        src_mask = create_padding_mask(
            src,
            pad_idx=self.pad_idx
        )
        x = self.embedding(src)
        x = x * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return x
