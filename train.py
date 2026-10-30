from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from dataset import TranslationDataset, collate_fn
from transformer import Transformer

DATA_DIR = Path("/home/yhy/yhy_study/my_transformer/data")
SRC_PATH = DATA_DIR / "train.en"
TGT_PATH = DATA_DIR / "train.fr"

BATCH_SIZE = 32
EPOCHS = 600
MAX_LEN = 5000

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEAD = 8
D_FF = 2048
DROPOUT = 0.1

WAEMUP_STEPS = 4000
PAD_IDX = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

if not SRC_PATH.exists() or not TGT_PATH.exists():
    raise FileExistsError(
        "未准备好文件"
    )

src_texts = SRC_PATH.read_text(
    encoding="utf-8"
).splitlines()
tgt_texts = TGT_PATH.read_text(
    encoding="utf-8"
).splitlines()

src_tokenizer = Tokenizer(min_freq=1)
tgt_tokenizer = Tokenizer(min_freq=1)
src_tokenizer.fit(src_texts)
tgt_tokenizer.fit(tgt_texts)

print("源词表大小: ", src_tokenizer.vocab_size)
print("目标词表大小: ", tgt_tokenizer.vocab_size)

DATA_DIR.mkdir(parents=True, exist_ok=True)
src_tokenizer.save(DATA_DIR / "src_tokenizer.json")
tgt_tokenizer.save(DATA_DIR / "tgt_tokenizer.json")

dataset = TranslationDataset(
    src_path=SRC_PATH,
    tgt_path=TGT_PATH,
    src_tokenizer=src_tokenizer,
    tgt_tokenizer=tgt_tokenizer,
    max_len=MAX_LEN
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(
        batch,
        src_pad_idx=src_tokenizer.pad_idx,
        tgt_pad_idx=tgt_tokenizer.pad_idx
    )
)

model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEAD,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    src_pad_idx=src_tokenizer.pad_idx,
    tgt_pad_idx=tgt_tokenizer.pad_idx
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9
)

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_tokenizer.pad_idx,
    label_smoothing=0.1
)

def noam_lr(step, d_model, warmup_steps):
    step = max(step, 1)

    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )

global_step = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        global_step += 1

        lr = noam_lr(
            global_step,
            D_MODEL,
            WAEMUP_STEPS
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} |"
        f"Loss: {avg_loss:.4f} | "
        f"LR: {lr:.8f}"
    )

    if (epoch + 1) % 50 == 0 or epoch + 1 == EPOCHS:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "d_model": D_MODEL,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEAD,
                "d_ff": D_FF,
                "src_vocab_size": src_tokenizer.vocab_size,
                "tgt_vocab_size": tgt_tokenizer.vocab_size,
            },
            "transformer_checkpoint.pt",
        )

print("训练结束，模型已保存。")
