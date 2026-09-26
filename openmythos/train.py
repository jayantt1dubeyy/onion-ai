"""CLI entry point: ``python -m openmythos.train --preset mini_60m --train-data ...``."""

from __future__ import annotations

import argparse

from .config import PRESETS
from .trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description="Train OpenMythos")
    ap.add_argument("--preset", default="mini_60m", choices=list(PRESETS))
    ap.add_argument("--train-data", nargs="+", default=[],
                    help=".txt files for training")
    ap.add_argument("--val-data", nargs="+", default=[],
                    help=".txt files for validation")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    mc, tc = PRESETS[args.preset]()
    if args.train_data:
        tc.train_files = args.train_data
    if args.val_data:
        tc.val_files = args.val_data
    tc.out_dir = args.out_dir
    if args.max_steps is not None:
        tc.max_steps = args.max_steps
    if not tc.train_files:
        raise SystemExit("no training data: pass --train-data file1.txt ...")

    Trainer(mc, tc).train(resume=args.resume)


if __name__ == "__main__":
    main()
