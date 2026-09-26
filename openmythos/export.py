"""Export a checkpoint to a Hugging Face-style directory.

Writes ``config.json`` (the ModelConfig), ``pytorch_model.bin`` (state dict)
and ``tokenizer.json`` (byte-level mapping) so the model can be loaded with
plain torch or inspected by other tools. Run: ``python -m openmythos.export
--ckpt checkpoints/final.pt --out hf_export/``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .config import ModelConfig
from .model import OpenMythosModel
from .tokenizer import ByteTokenizer


def export(ckpt_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_json(ckpt["model_cfg"])
    model = OpenMythosModel(mc)
    model.load_state_dict(ckpt["model"])

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(mc.to_json())
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    tok = ByteTokenizer()
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump({
            "type": "byte-level",
            "vocab_size": tok.vocab_size,
            "pad_id": tok.pad_id, "bos_id": tok.bos_id,
            "eos_id": tok.eos_id, "unk_id": tok.unk_id,
        }, f, indent=2)
    print(f"exported to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
