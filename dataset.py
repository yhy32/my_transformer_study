
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_path,
        tgt_path,
        src_tokenizer,
        tgt_tokenizer,
        max_len=None,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        src_lines = Path(src_path).read_text(
            encoding="utf-8"
        ).splitlines()

        tgt_lines = Path(tgt_path).read_text(
            encoding="utf-8"
        ).splitlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"源语料有 {len(src_lines)} 行，"
                f"目标语料有 {len(tgt_lines)} 行，数量不一致。"
            )

        self.pairs = [
            (src.strip(), tgt.strip())
            for src, tgt in zip(src_lines, tgt_lines)
            if src.strip() and tgt.strip()
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # 源序列：文本 + EOS
        src_ids = self.src_tokenizer.encode(
            src_text,
            add_bos=False,
            add_eos=True,
        )

        # 目标完整序列：BOS + 文本 + EOS
        tgt_ids = self.tgt_tokenizer.encode(
            tgt_text,
            add_bos=True,
            add_eos=True,
        )

        # 截断时保留 EOS
        if self.max_len is not None:
            if len(src_ids) > self.max_len:
                src_ids = src_ids[:self.max_len]
                src_ids[-1] = self.src_tokenizer.eos_idx

            if len(tgt_ids) > self.max_len:
                tgt_ids = tgt_ids[:self.max_len]
                tgt_ids[-1] = self.tgt_tokenizer.eos_idx

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    """
    batch: 多个 (src_ids, tgt_ids)

    返回：
        src_batch: (B, T_src)
        tgt_batch: (B, T_tgt)
    """
    src_sequences, tgt_sequences = zip(*batch)

    src_batch = pad_sequence(
        src_sequences,
        batch_first=True,
        padding_value=src_pad_idx,
    )

    tgt_batch = pad_sequence(
        tgt_sequences,
        batch_first=True,
        padding_value=tgt_pad_idx,
    )

    return src_batch, tgt_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tokenizer import Tokenizer

    # 创建临时小语料，独立测试 Dataset
    Path("toy.en").write_text(
        "I love you.\nHow are you?\n",
        encoding="utf-8",
    )
    Path("toy.fr").write_text(
        "Je t'aime.\nComment allez-vous ?\n",
        encoding="utf-8",
    )

    src_texts = Path("toy.en").read_text(
        encoding="utf-8"
    ).splitlines()
    tgt_texts = Path("toy.fr").read_text(
        encoding="utf-8"
    ).splitlines()

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()

    src_tokenizer.fit(src_texts)
    tgt_tokenizer.fit(tgt_texts)

    dataset = TranslationDataset(
        "toy.en",
        "toy.fr",
        src_tokenizer,
        tgt_tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            src_pad_idx=src_tokenizer.pad_idx,
            tgt_pad_idx=tgt_tokenizer.pad_idx,
        ),
    )

    src_batch, tgt_batch = next(iter(loader))

    print("src_batch:", src_batch)
    print("tgt_batch:", tgt_batch)
    print("src shape:", src_batch.shape)
    print("tgt shape:", tgt_batch.shape)

    assert src_batch.ndim == 2
    assert tgt_batch.ndim == 2
    assert src_batch.size(0) == tgt_batch.size(0)

    print("Dataset 测试通过！")

    Path("toy.en").unlink()
    Path("toy.fr").unlink()
