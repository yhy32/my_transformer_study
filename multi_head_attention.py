import torch
import torch.nn as nn

from attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, _ = x.shape

        x = x.reshape(B, T, self.num_heads, self.d_k)
        x = x.transpose(1, 2)

        return x

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        output, weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        output = output.transpose(1, 2).contiguous()

        B, Tq, _, _ = output.shape
        output = output.reshape(B, Tq, self.d_model)

        output = self.w_o(output)

        return output, weights