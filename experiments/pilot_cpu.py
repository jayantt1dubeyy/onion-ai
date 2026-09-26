"""Pilot experiment: 722K-param recurrent LM trained on CPU.

Proves the full pipeline end to end on real text (Gutenberg corpus):
data -> train -> eval perplexity -> checkpoint -> generate.

The real experiment (mini_60m on TinyStories) runs on a T4; see COLAB.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmythos.config import ModelConfig, TrainConfig
from openmythos.trainer import Trainer

mc = ModelConfig(
    vocab_size=512, max_seq_len=256,
    d_model=256, n_heads=8, n_kv_heads=4, d_ff=512,
    n_recurrent=4, grad_recurrence_steps=4,
    tie_word_embeddings=True,
)
tc = TrainConfig(
    train_files=["data/train.txt"], val_files=["data/val.txt"],
    seq_len=128, batch_size=8, grad_accum_steps=1,
    max_steps=3000, lr=1e-3, min_lr=1e-4, warmup_steps=150,
    weight_decay=0.1, mixed_precision=False,
    log_every=50, eval_every=500, save_every=1500,
    out_dir="checkpoints/pilot", seed=1337,
)

if __name__ == "__main__":
    print(f"pilot params: {__import__('openmythos.model', fromlist=['OpenMythosModel']).OpenMythosModel(mc).count_parameters():,}")
    Trainer(mc, tc, device="cpu").train()
