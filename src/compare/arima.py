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


def _fit_arima(y_train: np.ndarray, order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(y_train, order=order)
    # Keep it quiet; ARIMA on prices can trigger non-stationary warnings
    res = model.fit()
    return res


def _append_obs(res, y_new: float):
    # statsmodels provides append() for ARIMAResults in recent versions.
    if hasattr(res, "append"):
        return res.append([float(y_new)], refit=False)
    # Fallbacks for older APIs
    if hasattr(res, "extend"):
        return res.extend([float(y_new)])
    raise RuntimeError("statsmodels results object does not support append/extend for online updates")


def evaluate_arima(
    csv_path: str | None = None,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    order: tuple[int, int, int] = (5, 1, 0),
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
    total_len = len(y_raw)

    train_size = int(total_len * split_ratio)
    if train_size < max(order[0] + order[1] + order[2] + 5, 20):
        raise ValueError(f"Train size too small ({train_size}) for ARIMA{order}")

    y_train = y_raw[:train_size]

    res = _fit_arima(y_train, order=order)

    # pred[t] is the 1-step-ahead prediction for y[t]
    pred = np.full(total_len, np.nan, dtype=np.float64)

    # In-sample one-step-ahead predictions within training window.
    # For t in [1, train_size-1], predicted using data up to t-1.
    if train_size >= 2:
        ins = res.get_prediction(start=1, end=train_size - 1)
        pred[1:train_size] = np.asarray(ins.predicted_mean, dtype=np.float64).reshape(-1)

    # Out-of-sample rolling predictions with online state updates (no refit).
    cur = res
    for t in range(train_size, total_len):
        # at this point, cur has seen observations up to t-1
        yhat = float(cur.forecast(steps=1)[0])
        pred[t] = yhat
        cur = _append_obs(cur, y_raw[t])

    results = build_results_skeleton(
        method="arima",
        description=f"ARIMA baseline with rolling 1-step forecasts (fit on train only), order={order}.",
        csv_path=csv_path,
        split_ratio=split_ratio,
        val_ratio=val_ratio,
        config=config,
    )
    results["params"] = {"order": list(order)}

    for split in ("train", "val", "test"):
        ds = StockDataset(csv_path, config, mode=split, split_ratio=split_ratio, val_ratio=val_ratio)
        idxs = np.asarray(ds.valid_indices, dtype=np.int64)

        # Need predictions for y[idx+1]
        pred_raw = pred[idxs + 1]
        gt_raw = y_raw[idxs + 1]

        if np.isnan(pred_raw).any():
            raise RuntimeError(
                f"ARIMA produced NaN predictions for split={split}. "
                "Try a different order or ensure enough training data."
            )

        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, gt_raw),
            "n": int(len(ds)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIMA baseline (raw-scale evaluation)")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV (default: ModelConfig.DATA_CSV)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: src/compare/arima.json)",
    )
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--p", type=int, default=5, help="AR order")
    parser.add_argument("--d", type=int, default=1, help="Integration order")
    parser.add_argument("--q", type=int, default=0, help="MA order")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (Path(__file__).resolve().parent / "arima.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = evaluate_arima(
        csv_path=args.data,
        split_ratio=args.split_ratio,
        val_ratio=args.val_ratio,
        order=(int(args.p), int(args.d), int(args.q)),
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
