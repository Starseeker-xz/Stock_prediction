from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.compare._common import build_results_skeleton, ensure_project_root_on_path, load_series_raw, mse


def evaluate_ema(
    csv_path: str | None = None,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    span: int = 10,
    adjust: bool = False,
) -> dict:
    ensure_project_root_on_path()

    from src.config import ModelConfig
    from src.dataset import StockDataset

    config = ModelConfig

    if csv_path is None:
        csv_path = getattr(config, "DATA_CSV", os.path.join("data", "GOOGL_processed.csv"))

    feature_cols = list(getattr(config, "FEATURE_COLS", []))
    target_col = getattr(config, "TARGET_COL", None)
    if not target_col:
        raise ValueError("ModelConfig.TARGET_COL is not set")

    _, y_raw = load_series_raw(csv_path, feature_cols=feature_cols, target_col=target_col)

    # EMA computed using information up to time t; use EMA[t] to forecast y[t+1]
    import pandas as pd

    ema_series = (
        pd.Series(y_raw)
        .ewm(span=int(span), adjust=bool(adjust))
        .mean()
        .to_numpy(dtype=np.float64)
        .reshape(-1)
    )

    results = build_results_skeleton(
        method="ema",
        description=f"Exponential moving average forecast: pred[t+1] = EMA(y)[t] (span={span}, adjust={adjust}).",
        csv_path=csv_path,
        split_ratio=split_ratio,
        val_ratio=val_ratio,
        config=config,
    )
    results["params"] = {"span": int(span), "adjust": bool(adjust)}

    for split in ("train", "val", "test"):
        ds = StockDataset(csv_path, config, mode=split, split_ratio=split_ratio, val_ratio=val_ratio)
        idxs = np.asarray(ds.valid_indices, dtype=np.int64)

        pred_raw = ema_series[idxs]
        gt_raw = y_raw[idxs + 1]

        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, gt_raw),
            "n": int(len(ds)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA baseline (raw-scale evaluation)")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV (default: ModelConfig.DATA_CSV)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: src/compare/ema.json)",
    )
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--span", type=int, default=10, help="EMA span")
    parser.add_argument("--adjust", action="store_true", help="Use adjust=True for pandas ewm")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (Path(__file__).resolve().parent / "ema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = evaluate_ema(
        csv_path=args.data,
        split_ratio=args.split_ratio,
        val_ratio=args.val_ratio,
        span=args.span,
        adjust=args.adjust,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
