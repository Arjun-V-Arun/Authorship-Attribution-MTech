#!/usr/bin/env python3
"""
Run several experiments back-to-back, unattended.
Bank the cheap/robust jobs first; heavier fine-tunes last (they checkpoint,
so they can resume later if the 3 hours run out).

Each job: tees combined stdout/stderr to logs/<name>.log, records
start/end/duration/status to results/run_manifest.csv, and CONTINUES on failure.

Run (from repo root):  python3 src/run_experiments.py
Then just read logs/ and runs/ afterwards.
"""
import csv, os, subprocess, sys, time

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# (name, command) -- ordered by value-per-minute: frozen encoders first.
JOBS = [
    # 1-2: cheap frozen-encoder honest numbers (~10-15 min each), completes the table
    ("frozen_muril",
     [sys.executable, "src/dl_baseline.py", "--raw_dir", "data/raw",
      "--model", "google/muril-base-cased"]),
    ("frozen_indicbert",
     [sys.executable, "src/dl_baseline.py", "--raw_dir", "data/raw",
      "--model", "ai4bharat/indic-bert"]),
    # 3: the scientifically interesting one -- does fine-tuning beat char n-grams?
    ("finetune_distilmbert",
     [sys.executable, "src/train_finetune.py", "--raw_dir", "data/raw",
      "--model", "distilbert-base-multilingual-cased",
      "--epochs", "4", "--batch_size", "8", "--run_name", "distil_ft"]),
    # 4: optional, only if time remains (checkpoints -> resumable later)
    ("finetune_mbert",
     [sys.executable, "src/train_finetune.py", "--raw_dir", "data/raw",
      "--model", "bert-base-multilingual-cased",
      "--epochs", "3", "--batch_size", "8", "--run_name", "mbert_ft"]),
]

def main():
    manifest = "results/run_manifest.csv"
    new = not os.path.exists(manifest)
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if new:
            w.writerow(["job", "start", "end", "minutes", "status", "returncode"])
        for name, cmd in JOBS:
            print(f"\n{'='*60}\n>>> {name}\n{'='*60}", flush=True)
            start = time.time()
            logpath = os.path.join("logs", f"{name}.log")
            try:
                with open(logpath, "w", encoding="utf-8") as lf:
                    p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                rc, status = p.returncode, ("ok" if p.returncode == 0 else "FAILED")
            except Exception as e:
                rc, status = -1, f"EXC:{e}"
            mins = round((time.time() - start) / 60, 2)
            print(f"<<< {name}: {status} in {mins} min  (log: {logpath})", flush=True)
            w.writerow([name, time.strftime("%H:%M:%S", time.localtime(start)),
                        time.strftime("%H:%M:%S"), mins, status, rc])
            mf.flush()

    print("\nAll jobs attempted. See results/run_manifest.csv, logs/, runs/")

if __name__ == "__main__":
    main()
