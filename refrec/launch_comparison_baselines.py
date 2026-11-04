#!/usr/bin/env python3
"""Memory-aware three-seed launcher for the four comparison baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ("ml-100k", "Bookmarks", "LastFM", "datafoundation-user",
            "foursquare", "online_retail", "baidu_news_v2", "amazon_music")
SEEDS = (2023, 2024, 2025)
CORE = set(DATASETS[:4])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="results/comparison_baselines_8d3s_v1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_parallel", type=int, default=2)
    p.add_argument("--min_free_mb", type=int, default=9000)
    p.add_argument("--poll_seconds", type=int, default=20)
    p.add_argument("--train_samples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


def base_dir(dataset: str, seed: int) -> Path:
    if dataset in CORE:
        if seed == 2023:
            return Path("results/recal_mass_v1/base_runs_all_except_target") / dataset / f"{dataset}_v2_sasrec_base_all_except_target_seed2023"
        return Path("results/core_multiseed_v1/base_runs") / dataset / f"{dataset}_sasrec_base_seed{seed}"
    return Path("results/external_multiseed_v1/base_runs") / dataset / f"{dataset}_sasrec_base_seed{seed}"


def gpu_free_mb() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    return int(result.stdout.strip().splitlines()[0])


def main():
    args = parse_args(); root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    pending = []
    for dataset in DATASETS:
        for seed in SEEDS:
            base = base_dir(dataset, seed); out = root / dataset / f"seed{seed}"
            if (out / "test_metrics.json").exists():
                continue
            if not (base / "config.json").exists() or not (base / "best_model.pt").exists():
                raise FileNotFoundError(f"missing frozen checkpoint: {base}")
            cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
            if cfg.get("backbone") != "sasrec" or cfg.get("repeat_head") != "base" or cfg.get("seed") != seed:
                raise ValueError(f"unexpected frozen checkpoint config: {base}")
            pending.append((dataset, seed, base, out))
    active = []
    print(f"queue={len(pending)} max_parallel={args.max_parallel} min_free_mb={args.min_free_mb}", flush=True)
    while pending or active:
        survivors = []
        for job in active:
            code = job["process"].poll()
            if code is None:
                survivors.append(job); continue
            job["log"].close()
            print(f"FINISH dataset={job['dataset']} seed={job['seed']} exit={code}", flush=True)
            if code != 0:
                (job["out"] / "FAILED").write_text(f"exit_code={code}\n", encoding="utf-8")
        active = survivors
        while pending and len(active) < args.max_parallel:
            free = gpu_free_mb()
            if free < args.min_free_mb:
                print(f"WAIT free_mb={free} active={len(active)}", flush=True); break
            dataset, seed, base, out = pending.pop(0); out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "refrec.run_comparison_baselines",
                       "--base_run_dir", str(base), "--output_dir", str(out),
                       "--device", args.device, "--seed", str(seed),
                       "--train_samples", str(args.train_samples), "--epochs", str(args.epochs),
                       "--patience", str(args.patience)]
            log = (out / "run.log").open("a", encoding="utf-8")
            log.write("command=" + " ".join(command) + "\n"); log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append({"dataset": dataset, "seed": seed, "out": out,
                           "process": process, "log": log})
            print(f"START dataset={dataset} seed={seed} pid={process.pid} free_mb={free}", flush=True)
            time.sleep(3)
        if pending or active:
            time.sleep(args.poll_seconds)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
