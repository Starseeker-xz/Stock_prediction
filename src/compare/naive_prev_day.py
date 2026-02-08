import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _ensure_project_root_on_path() -> Path:
    """Allow running this file directly: `python src/compare/naive_prev_day.py`."""
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _mse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.float64).reshape(-1)
    gt = gt.astype(np.float64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    return float(np.mean((pred - gt) ** 2))


def _inverse_if_needed(values_1d: np.ndarray, scaler) -> np.ndarray:
    values_1d = values_1d.astype(np.float64).reshape(-1)
    if scaler is not None and (hasattr(scaler, "mean_") or hasattr(scaler, "min_")):
        return scaler.inverse_transform(values_1d.reshape(-1, 1)).reshape(-1)
    return values_1d


def evaluate_naive_prev_day(csv_path: str | None = None, split_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    _ensure_project_root_on_path()

    from src.config import ModelConfig
    from src.dataset import StockDataset

    config = ModelConfig

    if csv_path is None:
        csv_path = getattr(config, "DATA_CSV", os.path.join("data", "GOOGL_processed.csv"))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    results: dict[str, object] = {
        "method": "naive_prev_day",
        "description": "Predict y[t+1] with y[t] (previous day target).",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_csv": csv_path,
        "split": {"train_ratio": split_ratio, "val_ratio": val_ratio},
        "config": {
            "FEATURE_COLS": list(getattr(config, "FEATURE_COLS", [])),
            "TARGET_COL": getattr(config, "TARGET_COL", None),
            "TARGET_NORMALIZE": bool(getattr(config, "TARGET_NORMALIZE", True)),
            "FEATURE_NORMALIZE_MASK": list(getattr(config, "FEATURE_NORMALIZE_MASK", [])),
            "SEQ_LEN_LONG": int(getattr(config, "SEQ_LEN_LONG", 0)),
            "SEQ_LEN_MEDIUM": int(getattr(config, "SEQ_LEN_MEDIUM", 0)),
            "SEQ_LEN_SHORT": int(getattr(config, "SEQ_LEN_SHORT", 0)),
        },
        "metrics": {},
    }

    # Fit scaler on train split (StockDataset does this internally)
    ds_train = StockDataset(csv_path, config, mode="train", split_ratio=split_ratio, val_ratio=val_ratio)
    target_scaler = getattr(ds_train, "target_scaler", None)

    for split in ("train", "val", "test"):
        ds = StockDataset(csv_path, config, mode=split, split_ratio=split_ratio, val_ratio=val_ratio)
        idxs = np.asarray(ds.valid_indices, dtype=np.int64)

        pred_scaled = ds.target[idxs, 0]
        gt_scaled = ds.target[idxs + 1, 0]

        pred_raw = _inverse_if_needed(pred_scaled, target_scaler)
        gt_raw = _inverse_if_needed(gt_scaled, target_scaler)

        results["metrics"][split] = {
            "mse_raw": _mse(pred_raw, gt_raw),
            "n": int(len(ds)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive baseline: predict next day target by previous day target")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV (default: ModelConfig.DATA_CSV)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: src/compare/naive_prev_day.json)",
    )
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (Path(__file__).resolve().parent / "naive_prev_day.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = evaluate_naive_prev_day(args.data, split_ratio=args.split_ratio, val_ratio=args.val_ratio)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
