import re
import json
from collections import Counter

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

SPECIAL_TOKEN = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

class Tokenizer:

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.token_to_id = {
            token:idx
            for idx, token in enumerate(SPECIAL_TOKEN) 
        }
        self.id_to_token = list(SPECIAL_TOKEN)

    @property
    def pad_idx(self):
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_idx(self):
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_idx(self):
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_idx(self):
        return self.token_to_id[EOS_TOKEN]

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    def fit(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(tokenize(text))

        tokens = [
            token
            for token, count in counter.most_common()
            if count >= self.min_freq
            and token not in self.token_to_id
        ]

        for token in tokens:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(self, text, add_bos=False, add_eos=True):
        tokens = tokenize(text)
        ids = [
            self.token_to_id.get(token, self.unk_idx)
            for token in tokens
        ]

        if add_bos:
            ids.insert(0, self.bos_idx)

        if add_eos:
            ids.append(self.eos_idx)

        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []

        for idx in ids:
            token = self.id_to_token[int(idx)]

            if skip_special_tokens and token in SPECIAL_TOKEN:
                continue

            tokens.append(token)

        text = " ".join(tokens)
        text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)
        text = re.sub(r"\s+([\)\]\}])", r"\1", text)

        return text

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_freq": self.min_freq,
                    "id_to_token": self.id_to_token,
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(min_freq=data["min_freq"])
        tokenizer.id_to_token = data["id_to_token"]
        tokenizer.token_to_id = {
            token: idx
            for idx, token in enumerate(tokenizer.id_to_token)
        }
        return tokenizer

if __name__ == "__main__":
    texts = [
        "I love cats!",
        "I love dogs.",
        "You love cats too!"
    ]

    tokenizer = Tokenizer(min_freq=1)
    tokenizer.fit(texts)

    sentence = "I love cats!"
    ids = tokenizer.encode(
        sentence,
        add_bos=True,
        add_eos=True
    )

    print(tokenizer.vocab_size)
    print(sentence)
    print(ids)
    print(tokenizer.decode(ids))