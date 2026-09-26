"""Text data pipeline.

``TextDataset`` reads raw ``.txt`` files, tokenizes them once, and serves
fixed-length chunks of ``seq_len + 1`` tokens; the trainer shifts these into
inputs/targets. Token streams from multiple files are concatenated so no
tokens are wasted at file boundaries. Short files are fine -- chunks are
sampled uniformly from the concatenated stream.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class TextDataset(Dataset):
    def __init__(self, files: list[str], tokenizer: ByteTokenizer,
                 seq_len: int) -> None:
        self.seq_len = seq_len
        self.tok = tokenizer
        stream: list[int] = []
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                stream.extend(tokenizer.encode(fh.read()))
        if len(stream) < seq_len + 1:
            raise ValueError(
                f"only {len(stream)} tokens in {files}; need > {seq_len}"
            )
        self.data = torch.tensor(stream, dtype=torch.long)
        self.n_chunks = (len(stream) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, i: int) -> torch.Tensor:
        s = (i * self.seq_len) % (len(self.data) - self.seq_len)
        return self.data[s:s + self.seq_len + 1]


def make_loader(files: list[str], tokenizer: ByteTokenizer, seq_len: int,
                batch_size: int, shuffle: bool, seed: int,
                num_workers: int = 0):
    """DataLoader with a seeded generator for reproducible shuffling."""
    from torch.utils.data import DataLoader
    ds = TextDataset(files, tokenizer, seq_len)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=gen, num_workers=num_workers, drop_last=True)
