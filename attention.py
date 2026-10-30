import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v

    return output, weights
