import torch

def create_padding_mask(src, pad_idx=0):
    return (src != pad_idx)[:, None, None, :]

def create_causal_mask(seq_len, device=None):
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )

    return mask[None, None, :, :]

def create_decoder_self_attention_mask(tgt, pad_idx=0):
    padding_mask = create_padding_mask(tgt, pad_idx)
    causal_mask = create_causal_mask(
        tgt.size(1), device=tgt.device
    )
    return padding_mask & causal_mask

if __name__ == "__main__":
    src = torch.tensor([
        [4, 5, 6, 8],
        [6, 7, 8, 0]
    ])

    mask = create_padding_mask(src, 0)

    print(mask.shape)
    print(mask)