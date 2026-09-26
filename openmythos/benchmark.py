"""Sanity checks and hardware estimates.

- ``--mode params``: exact parameter count of a preset (no torch needed
  beyond the model definition... actually builds the model, so torch needed).
- ``--mode train``: runs a short training smoke test (forward + backward +
  optimizer step) and reports tokens/sec -- run this on the T4 before any
  long job.
- ``--mode vram``: estimates training VRAM for a preset (analytic: weights,
  gradients, AdamW states in fp32, plus a rough activation term).
"""

from __future__ import annotations

import argparse
import time

import torch

from .config import PRESETS
from .model import OpenMythosModel


def count_params(preset: str) -> int:
    mc, _ = PRESETS[preset]()
    return OpenMythosModel(mc).count_parameters()


def estimate_vram_gb(preset: str) -> dict:
    """Analytic VRAM estimate for training in bf16 with AdamW (fp32 states)."""
    mc, tc = PRESETS[preset]()
    p = count_params(preset)
    weights = p * 2          # bf16 params
    grads = p * 2            # bf16 grads
    adam = p * 8             # fp32 m + v
    # Activations: very rough -- batch * seq * d_model * layers-equivalent
    # (n_recurrent iterations each store ~12 tensors for backward).
    act = tc.batch_size * tc.seq_len * mc.d_model * mc.n_recurrent * 12 * 2
    total = weights + grads + adam + act
    return {
        "params": p,
        "weights_gb": weights / 1e9,
        "grads_gb": grads / 1e9,
        "adam_gb": adam / 1e9,
        "activations_gb": act / 1e9,
        "total_gb": total / 1e9,
    }


def smoke_train(preset: str, steps: int = 5, device: str | None = None) -> dict:
    """Forward + backward + optimizer step on random data; reports tok/s."""
    from .optimizer import build_optimizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mc, tc = PRESETS[preset]()
    torch.manual_seed(0)
    model = OpenMythosModel(mc).to(device)
    opt = build_optimizer(model, tc)
    model.train()

    B, L = tc.batch_size, tc.seq_len
    ids = torch.randint(0, mc.vocab_size, (B, L + 1), device=device)
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = model.forward_loss(ids)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    toks = steps * B * L
    return {"tokens": toks, "seconds": round(dt, 2),
            "tokens_per_sec": round(toks / dt, 1), "loss": round(loss.item(), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny_debug",
                    choices=list(PRESETS))
    ap.add_argument("--mode", default="params",
                    choices=["params", "vram", "train"])
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "params":
        print(f"{args.preset}: {count_params(args.preset):,} parameters")
    elif args.mode == "vram":
        est = estimate_vram_gb(args.preset)
        print(f"{args.preset}: {est['params']:,} parameters")
        for k in ["weights_gb", "grads_gb", "adam_gb", "activations_gb",
                  "total_gb"]:
            print(f"  {k}: {est[k]:.2f} GB")
    elif args.mode == "train":
        r = smoke_train(args.preset, args.steps)
        print(f"{args.preset}: {r['tokens_per_sec']} tok/s "
              f"({r['tokens']} tokens in {r['seconds']}s), loss {r['loss']}")


if __name__ == "__main__":
    main()
