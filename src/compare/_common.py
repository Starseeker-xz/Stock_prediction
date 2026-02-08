from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_project_root_on_path() -> Path:
    """Allow running compare scripts directly: `python src/compare/x.py`."""
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    return float(np.mean((pred - gt) ** 2))


def load_series_raw(csv_path: str, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load data in the same row-filtering way as StockDataset.

    StockDataset drops rows with NaN in feature columns only.
    We mimic that so indices align.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if target_col not in df.columns:
        raise ValueError(f"TARGET_COL '{target_col}' not found in CSV columns")

    y = df[target_col].to_numpy(dtype=np.float64).reshape(-1)
    if np.isnan(y).any():
        raise ValueError("TARGET_COL contains NaN after dropna(subset=FEATURE_COLS); data may be inconsistent")

    return df, y


def build_results_skeleton(method: str, description: str, csv_path: str, split_ratio: float, val_ratio: float, config) -> dict:
    return {
        "method": method,
        "description": description,
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
        "params": {},
    }
