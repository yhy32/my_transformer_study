import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from masking import create_padding_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_len=5000, dropout=0.1, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=src_pad_idx
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=tgt_pad_idx
        )

        self.output_projection = nn.Linear(
            d_model, tgt_vocab_size
        )

    def forward(self, src, tgt):

        src_mask = create_padding_mask(
            src, pad_idx=self.src_pad_idx
        )

        enc_output = self.encoder(src)

        dec_output = self.decoder(
            tgt,
            enc_output,
            src_mask=src_mask
        )

        logits = self.output_projection(dec_output)

        return logits

if __name__ == "__main__":
    torch.manual_seed(42)

    src_vocab_size = 100
    tgt_vocab_size = 120

    src = torch.tensor(
        [
            [4, 8, 2, 9, 3],
            [6, 7, 5, 0, 0]
        ]
    )

    tgt = torch.tensor(
        [
            [1, 5, 8, 2],
            [1, 7, 9, 0]
        ]
    )

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        64,
        2,
        4,
        128,
        100,
        0.1,
        0,
        0
    )

    model.eval()

    with torch.no_grad():
        logits = model(src, tgt)

    print(src.shape)
    print(tgt.shape)
    print(logits.shape)

    model.train()
    logits = model(src, tgt)
    loss = logits.mean()
    loss.backward()

    assert model.output_projection.weight.grad is not None

    print("完成反向传播")