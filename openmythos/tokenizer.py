"""Byte-level tokenizer: no training step, works on any text.

Vocabulary = 256 byte values + special tokens. Bytes are a universal,
language-agnostic segmentation; the trade-off is longer sequences than BPE,
which is acceptable for a research-scale model.
"""

from __future__ import annotations


class ByteTokenizer:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self) -> None:
        self.specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        # ids 0..255 -> raw bytes; ids 256..259 -> specials
        self.pad_id = 256
        self.bos_id = 257
        self.eos_id = 258
        self.unk_id = 259
        self.vocab_size = 260

    # -- encoding ---------------------------------------------------------
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(b for b in ids if b < 256)
        return raw.decode("utf-8", errors="replace")

    def batch_encode(self, texts: list[str], max_len: int,
                     pad_to_max: bool = True) -> list[list[int]]:
        out = []
        for t in texts:
            ids = self.encode(t)[:max_len]
            if pad_to_max:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            out.append(ids)
        return out
