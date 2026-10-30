import torch

from tokenizer import Tokenizer
from transformer import Transformer

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def load_model(
        checkpoint_path="transformer_checkpoint.pt",
        src_tokenizer_path="/home/yhy/yhy_study/my_transformer/data/src_tokenizer.json",
        tgt_tokenizer_path="/home/yhy/yhy_study/my_transformer/data/tgt_tokenizer.json"
):
    src_tokenizer = Tokenizer.load(src_tokenizer_path)
    tgt_tokenizer = Tokenizer.load(tgt_tokenizer_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model = Transformer(
        src_vocab_size=checkpoint["src_vocab_size"],
        tgt_vocab_size=checkpoint["tgt_vocab_size"],
        d_model=checkpoint["d_model"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        d_ff=checkpoint["d_ff"],
        max_len=5000,
        dropout=0.1,
        src_pad_idx=src_tokenizer.pad_idx,
        tgt_pad_idx=tgt_tokenizer.pad_idx,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, src_tokenizer, tgt_tokenizer

@torch.no_grad()
def translate(
    model, 
    sentence,
    src_tokenizer,
    tgt_tokenizer,
    max_new_tokens=100
):
    src_ids = src_tokenizer.encode(
        sentence,
        add_bos=False,
        add_eos=True
    )
    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated = torch.tensor(
        [[tgt_tokenizer.bos_idx]],
        dtype=torch.long,
        device=DEVICE,
    )


    # 3. 每次预测一个新 token
    for _ in range(max_new_tokens):
        logits = model(src, generated)

        # 取最后一个位置的词表分数
        next_token_logits = logits[:, -1, :]

        # Greedy Decoding：选择分数最大的 token
        next_token = next_token_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        # 追加到目标序列
        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        # 生成 EOS 后结束
        if next_token.item() == tgt_tokenizer.eos_idx:
            break

    # 去掉 BOS、EOS 等特殊 token，转回文本
    result = tgt_tokenizer.decode(
        generated[0].tolist(),
        skip_special_tokens=True,
    )

    return result


if __name__ == "__main__":
    model, src_tokenizer, tgt_tokenizer = load_model()

    while True:
        sentence = input("请输入英语句子（输入 q 退出）：").strip()

        if sentence.lower() == "q":
            break

        if not sentence:
            continue

        result = translate(
            model,
            sentence,
            src_tokenizer,
            tgt_tokenizer,
        )

        print("翻译结果:", result)
  