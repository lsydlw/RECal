"""FARRec 通用工具：进度条、运行目录、日志、数据集默认配置与消融变体。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def tqdm_bar(iterable, *, disable: bool = False, **kwargs):
    from tqdm import tqdm
    return tqdm(iterable, disable=disable, file=sys.__stderr__, dynamic_ncols=True, **kwargs)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in vars(args).items():
        out[key] = str(val) if isinstance(val, Path) else val
    out["command"] = " ".join(sys.argv)
    return out


def resolve_run_dir(output_root: str | Path, dataset: str, run_name: str) -> Path:
    path = Path(output_root) / dataset / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_ablation_run_dir(
    output_root: str | Path, dataset: str, variant: str, seed: int,
) -> Path:
    path = Path(output_root) / dataset / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


class TeeLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("w", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "ml-100k": {
        "main_topk": 1, "min_main_count": 2, "cat_smooth_k": 2.0,
        "gamma_weak": 0.8, "gamma_repeat": 0.8, "lambda_weak_aux": 0.03,
        "router_fixed_weights": [0.35, 0.35, 0.15, 0.15],
    },
    "baidu_news": {
        "main_topk": 2, "min_main_count": 2, "cat_smooth_k": 2.0,
        "gamma_weak": 0.7, "gamma_repeat": 0.9, "lambda_weak_aux": 0.01,
        "router_fixed_weights": [0.30, 0.30, 0.10, 0.30],
    },
    "datafoundation-user": {
        "main_topk": 3, "min_main_count": 1, "cat_smooth_k": 5.0,
        "gamma_weak": 0.75, "gamma_repeat": 0.8, "lambda_weak_aux": 0.02,
        "router_fixed_weights": [0.30, 0.35, 0.20, 0.15],
    },
    "pijiu": {
        "main_topk": 2, "min_main_count": 2, "cat_smooth_k": 2.0,
        "gamma_weak": 0.85, "gamma_repeat": 0.8, "lambda_weak_aux": 0.03,
    },
    "Bookmarks": {
        "main_topk": 2, "min_main_count": 2, "cat_smooth_k": 3.0,
        "gamma_weak": 0.8, "gamma_repeat": 0.75, "lambda_weak_aux": 0.02,
    },
    "LastFM": {
        "main_topk": 2, "min_main_count": 2, "cat_smooth_k": 3.0,
        "gamma_weak": 0.8, "gamma_repeat": 0.85, "lambda_weak_aux": 0.02,
    },
    "foursquare": {
        "main_topk": 1, "min_main_count": 2, "gamma_repeat": 0.8,
    },
    "baidu_news_v2": {
        "main_topk": 2, "min_main_count": 2, "gamma_repeat": 0.9,
    },
    "amazon_music": {
        "main_topk": 1, "min_main_count": 2, "gamma_repeat": 0.8,
    },
    "online_retail": {
        "main_topk": 1, "min_main_count": 2, "gamma_repeat": 0.8,
    },
}

SUPPORTED_DATASETS = list(DATASET_DEFAULTS.keys())

VARIANT_FLAGS: dict[str, dict[str, Any]] = {
    "full": {
        "disable_weak_expert": False, "disable_repeat_expert": False,
        "disable_router": False, "disable_recent_expert": False,
        "disable_weak_aux": False,
    },
    "w_o_weak_expert": {
        "disable_weak_expert": True, "disable_repeat_expert": False,
        "disable_router": False, "disable_recent_expert": False,
        "disable_weak_aux": False,
    },
    "w_o_repeat_expert": {
        "disable_weak_expert": False, "disable_repeat_expert": True,
        "disable_router": False, "disable_recent_expert": False,
        "disable_weak_aux": False,
    },
    "w_o_router": {
        "disable_weak_expert": False, "disable_repeat_expert": False,
        "disable_router": True, "disable_recent_expert": False,
        "disable_weak_aux": False,
    },
    "w_o_recent_expert": {
        "disable_weak_expert": False, "disable_repeat_expert": False,
        "disable_router": False, "disable_recent_expert": True,
        "disable_weak_aux": False,
    },
    "w_o_weak_aux": {
        "disable_weak_expert": False, "disable_repeat_expert": False,
        "disable_router": False, "disable_recent_expert": False,
        "disable_weak_aux": True,
    },
}


def apply_dataset_defaults(parser: argparse.ArgumentParser, dataset: str | None) -> None:
    if dataset is None or dataset not in DATASET_DEFAULTS:
        return
    parser.set_defaults(**DATASET_DEFAULTS[dataset])


def apply_variant(args: argparse.Namespace) -> argparse.Namespace:
    variant = getattr(args, "variant", "full")
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"unknown variant: {variant}")
    for key, val in VARIANT_FLAGS[variant].items():
        setattr(args, key, val)
    if args.disable_weak_aux:
        args.lambda_weak_aux = 0.0
    return args


def ablation_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "variant": args.variant,
        "disable_weak_expert": args.disable_weak_expert,
        "disable_repeat_expert": args.disable_repeat_expert,
        "disable_router": args.disable_router,
        "disable_recent_expert": args.disable_recent_expert,
        "disable_weak_aux": args.disable_weak_aux,
        "gamma_weak": args.gamma_weak,
        "gamma_repeat": args.gamma_repeat,
        "alpha_main": args.alpha_main,
        "alpha_recent": args.alpha_recent,
        "cat_smooth_k": args.cat_smooth_k,
        "lambda_weak_aux": args.lambda_weak_aux,
        "router_fixed_weights": args.router_fixed_weights,
    }
