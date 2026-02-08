from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.compare._common import build_results_skeleton, ensure_project_root_on_path, load_series_raw, mse


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def evaluate_var(
    csv_path: str | None = None,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    maxlags: int = 15,
    ic: str = "aic",
    lags: int | None = None,
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

    # Align row filtering with StockDataset
    df, y_raw = load_series_raw(csv_path, feature_cols=feature_cols, target_col=target_col)

    var_cols = _unique_preserve_order(feature_cols + [target_col])
    for c in var_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")

    X_raw = df[var_cols].to_numpy(dtype=np.float64)
    if np.isnan(X_raw).any():
        raise ValueError("VAR input contains NaN after dropna(subset=FEATURE_COLS); please check preprocessing")

    total_len = X_raw.shape[0]
    train_size = int(total_len * split_ratio)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_raw[:train_size])
    X_scaled = scaler.transform(X_raw)

    from statsmodels.tsa.api import VAR

    if lags is None:
        selector = VAR(X_train_scaled).select_order(maxlags=int(maxlags))
        selected = selector.selected_orders.get(str(ic).lower())
        if selected is None:
            # fallback to 5
            selected = 5
        k_ar = int(max(1, selected))
    else:
        k_ar = int(lags)
        if k_ar < 1:
            raise ValueError("lags must be >= 1")

    model = VAR(X_train_scaled)
    res = model.fit(k_ar)

    # pred_target[t] predicts y_raw[t] one-step-ahead
    pred_target = np.full(total_len, np.nan, dtype=np.float64)

    # In-sample 1-step ahead within training window (parameters fit on all train)
    # Use res.fittedvalues which corresponds to in-sample predictions for t=k_ar..train_size-1
    fitted = np.asarray(res.fittedvalues, dtype=np.float64)
    # fitted is aligned to X_train_scaled[k_ar:]
    target_idx = var_cols.index(target_col)
    for t in range(k_ar, train_size):
        pred_target[t] = fitted[t - k_ar, target_idx]

    # Rolling out-of-sample using fixed parameters and updated history
    history = X_scaled[:train_size].tolist()
    for t in range(train_size, total_len):
        hist_arr = np.asarray(history[-k_ar:], dtype=np.float64)
        yhat_vec = res.forecast(y=hist_arr, steps=1)[0]
        pred_target[t] = float(yhat_vec[target_idx])
        history.append(X_scaled[t].tolist())

    # Inverse scale target back to raw
    # StandardScaler inverse: x_raw = x_scaled * scale + mean
    target_mean = float(scaler.mean_[target_idx])
    target_scale = float(scaler.scale_[target_idx])
    pred_raw_series = pred_target * target_scale + target_mean

    results = build_results_skeleton(
        method="var",
        description=f"VAR baseline with rolling 1-step forecasts (fit on train only), lags={k_ar}.",
        csv_path=csv_path,
        split_ratio=split_ratio,
        val_ratio=val_ratio,
        config=config,
    )
    results["params"] = {
        "var_cols": var_cols,
        "lags": int(k_ar),
        "lags_selection": None if lags is not None else {"maxlags": int(maxlags), "ic": str(ic).lower()},
        "scaler": "StandardScaler(train_only)",
    }

    for split in ("train", "val", "test"):
        ds = StockDataset(csv_path, config, mode=split, split_ratio=split_ratio, val_ratio=val_ratio)
        idxs = np.asarray(ds.valid_indices, dtype=np.int64)

        pred_raw = pred_raw_series[idxs + 1]
        gt_raw = y_raw[idxs + 1]

        if np.isnan(pred_raw).any():
            raise RuntimeError(
                f"VAR produced NaN predictions for split={split}. "
                "Try a smaller lags/maxlags or check data." 
            )

        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, gt_raw),
            "n": int(len(ds)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VAR baseline (raw-scale evaluation)")
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV (default: ModelConfig.DATA_CSV)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: src/compare/var.json)",
    )
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--maxlags", type=int, default=15, help="Max lags for order selection")
    parser.add_argument("--ic", type=str, default="aic", choices=["aic", "bic", "hqic", "fpe"], help="Information criterion")
    parser.add_argument("--lags", type=int, default=None, help="Set fixed lags (overrides selection)")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (Path(__file__).resolve().parent / "var.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = evaluate_var(
        csv_path=args.data,
        split_ratio=args.split_ratio,
        val_ratio=args.val_ratio,
        maxlags=args.maxlags,
        ic=args.ic,
        lags=args.lags,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
