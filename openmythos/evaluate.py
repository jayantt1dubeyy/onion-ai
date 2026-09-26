"""Standalone evaluation: perplexity of a checkpoint on a text file."""

from __future__ import annotations

import argparse
import math

import torch

from .config import ModelConfig
from .dataset import make_loader
from .model import OpenMythosModel
from .tokenizer import ByteTokenizer


def perplexity(ckpt_path: str, text_file: str, seq_len: int = 1024,
               batch_size: int = 4, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OpenMythosModel(ModelConfig.from_json(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = make_loader([text_file], ByteTokenizer(), seq_len, batch_size,
                         shuffle=False, seed=0)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, ce, _ = model.forward_loss(batch)
            total += ce.item() * batch.shape[0]
            n += batch.shape[0]
    return math.exp(total / max(1, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    ppl = perplexity(args.ckpt, args.text, args.seq_len, args.batch_size)
    print(f"perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
