"""Training loop for OpenMythos.

Features: gradient accumulation, bfloat16 autocast on CUDA, gradient
clipping, periodic validation perplexity, checkpoint save/resume (model,
optimizer, scheduler step, RNG states), and a JSONL loss log for plotting.

Checkpoints are plain ``torch.save`` dicts: ``checkpoints/step_{N}.pt`` plus
a ``latest.pt`` symlink-equivalent (a copied file, for filesystems without
symlink support).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time

import torch

from .config import ModelConfig, TrainConfig
from .dataset import make_loader
from .model import OpenMythosModel
from .optimizer import build_optimizer
from .scheduler import WarmupCosine
from .tokenizer import ByteTokenizer


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 device: str | None = None) -> None:
        self.mc, self.tc = model_cfg, train_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(train_cfg.seed)

        self.tok = ByteTokenizer()
        assert self.tok.vocab_size <= model_cfg.vocab_size, (
            "model vocab_size smaller than tokenizer output range"
        )
        self.model = OpenMythosModel(model_cfg).to(self.device)
        self.opt = build_optimizer(self.model, train_cfg)
        self.sched = WarmupCosine(self.opt, train_cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.mixed_precision
                                           and self.device == "cuda")
        self.step = 0
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        self.log_path = os.path.join(train_cfg.out_dir, "loss.jsonl")

    # -- checkpointing ----------------------------------------------------
    def save(self, tag: str | None = None) -> str:
        tag = tag or f"step_{self.step}"
        path = os.path.join(self.tc.out_dir, f"{tag}.pt")
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
            "model_cfg": self.mc.to_json(),
            "train_cfg": self.tc.to_json(),
        }, path)
        shutil.copyfile(path, os.path.join(self.tc.out_dir, "latest.pt"))
        return path

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng"])
        if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        self.step = ckpt["step"]
        self.sched.step(self.step)

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in val_loader:
            batch = batch.to(self.device)
            _, ce, _ = self.model.forward_loss(batch)
            total += ce.item() * batch.shape[0]
            n += batch.shape[0]
        self.model.train()
        return math.exp(total / max(1, n))

    # -- main loop ----------------------------------------------------------
    def train(self, resume: str | None = None) -> None:
        if resume:
            self.load(resume)
            print(f"resumed from {resume} at step {self.step}")

        train_loader = make_loader(self.tc.train_files, self.tok, self.tc.seq_len,
                                   self.tc.batch_size, shuffle=True,
                                   seed=self.tc.seed + self.step)
        val_loader = (make_loader(self.tc.val_files, self.tok, self.tc.seq_len,
                                  self.tc.batch_size, shuffle=False,
                                  seed=self.tc.seed)
                      if self.tc.val_files else None)

        self.model.train()
        use_amp = self.tc.mixed_precision and self.device == "cuda"
        t0 = time.time()
        log = open(self.log_path, "a")
        data_iter = iter(train_loader)

        while self.step < self.tc.max_steps:
            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0
            for _ in range(self.tc.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = batch.to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=use_amp):
                    loss, ce, aux = self.model.forward_loss(batch)
                    loss = loss / self.tc.grad_accum_steps
                self.scaler.scale(loss).backward()
                acc_loss += ce.item()

            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.tc.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            lr = self.sched.step(self.step)
            self.step += 1

            if self.step % self.tc.log_every == 0:
                dt = time.time() - t0
                rec = {"step": self.step,
                       "loss": acc_loss / self.tc.grad_accum_steps,
                       "lr": lr, "secs": round(dt, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"step {self.step:6d} | loss {rec['loss']:.4f} | "
                      f"lr {lr:.2e} | {dt:.0f}s", flush=True)

            if val_loader and self.step % self.tc.eval_every == 0:
                ppl = self.evaluate(val_loader)
                print(f"  [eval] step {self.step} val ppl {ppl:.2f}", flush=True)
                log.write(json.dumps({"step": self.step, "val_ppl": ppl}) + "\n")
                log.flush()

            if self.step % self.tc.save_every == 0:
                path = self.save()
                print(f"  [ckpt] saved {path}", flush=True)

        log.close()
        self.save("final")
        print("training complete")
