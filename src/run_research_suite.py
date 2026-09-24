#!/usr/bin/env python3
"""
Convenience launcher for the Malayalam AA experimental suite.

Run from repository root:

  python src/run_research_suite.py --stage all

Stages:
  neural       CNN + BiLSTM + BiGRU
  transformer  MuRIL work-level fine-tuning
  ablations    document length + author-count scaling
  all          all three

The defaults are intentionally conservative for an RTX 4070-class laptop.
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 90)
    print("RUNNING:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def main(args):
    py = sys.executable
    src = os.path.join(args.root, "src")

    if args.stage in {"neural", "all"}:
        run([
            py, os.path.join(src, "neural_models.py"),
            "--root", args.root,
            "--texts", "texts_strict",
            "--epochs", str(args.neural_epochs),
            "--repeats", str(args.repeats),
            "--batch-size", str(args.neural_batch),
            "--max-len", str(args.neural_max_len),
        ])

    if args.stage in {"transformer", "all"}:
        run([
            py, os.path.join(src, "transformer_malayalam.py"),
            "--root", args.root,
            "--texts", "texts_strict",
            "--model", args.model,
            "--epochs", str(args.transformer_epochs),
            "--repeats", str(args.transformer_repeats),
            "--batch-size", str(args.transformer_batch),
            "--grad-accum", str(args.grad_accum),
            "--max-length", "512",
            "--stride", "128",
        ])

    if args.stage in {"ablations", "all"}:
        run([
            py, os.path.join(src, "ablations_malayalam.py"),
            "--root", args.root,
            "--texts", "texts_strict",
            "--repeats", str(args.repeats),
            "--author-repeats", str(args.author_repeats),
        ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--stage",
        choices=["neural", "transformer", "ablations", "all"],
        default="all"
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--neural-epochs", type=int, default=12)
    ap.add_argument("--neural-batch", type=int, default=8)
    ap.add_argument("--neural-max-len", type=int, default=1200)
    ap.add_argument("--transformer-model",
                    dest="model", default="google/muril-base-cased")
    ap.add_argument("--transformer-epochs", type=int, default=3)
    ap.add_argument("--transformer-repeats", type=int, default=1)
    ap.add_argument("--transformer-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--author-repeats", type=int, default=5)
    main(ap.parse_args())
