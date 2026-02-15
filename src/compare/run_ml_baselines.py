"""ML baselines — 配置与运行入口.

用法:
    python src/compare/run_ml_baselines.py            # 运行全部 5 个模型
    python src/compare/run_ml_baselines.py --models linear_regression random_forest
    python src/compare/run_ml_baselines.py --window 10 --output results/ml.json

运行结束后会在终端打印汇总表格，并将详细结果写入 JSON 文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 环境准备
# ---------------------------------------------------------------------------

def _ensure_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_ensure_project_root()


# ===================================================================
# 配置区  ——  在此修改各模型的超参数
# ===================================================================

# 通用
SPLIT_RATIO = 0.7       # 训练集比例
VAL_RATIO   = 0.15      # 验证集比例
WINDOW_SIZE = 5          # 滑窗大小（天数）
SEED        = 56         # 全局随机种子

# ---------- Linear Regression ----------
LINEAR_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
}

# ---------- Polynomial Regression ----------
POLY_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
    "degree": 2,
}

# ---------- Random Forest ----------
RF_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
    "n_estimators": 200,
    "max_depth": 12,
    "seed": SEED,
}

# ---------- SVM (SVR) ----------
SVM_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.1,
}

# ---------- CNN (1D Conv) ----------
CNN_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
    "channels": [32, 64, 64],
    "kernel_size": 3,
    "dropout": 0.0,
    "lr": 1e-4,
    "epochs": 200,
    "batch_size": 32,
    "patience": 40,
    "seed": SEED,
}

# ---------- Simple LSTM ----------
LSTM_PARAMS: dict = {
    "window_size": WINDOW_SIZE,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.0,
    "lr": 1e-4,
    "weight_decay": 0,
    "epochs": 200,
    "batch_size": 32,
    "patience": 40,
    "seed": SEED,
}

# 注册表: 名称 → (evaluate 函数名, 参数字典)
MODEL_REGISTRY: dict[str, tuple[str, dict]] = {
    "linear_regression":     ("evaluate_linear_regression",     LINEAR_PARAMS),
    "polynomial_regression": ("evaluate_polynomial_regression", POLY_PARAMS),
    "random_forest":         ("evaluate_random_forest",         RF_PARAMS),
    "svm":                   ("evaluate_svm",                   SVM_PARAMS),
    "cnn_1d":                ("evaluate_cnn1d",                 CNN_PARAMS),
    "lstm":                  ("evaluate_lstm",                  LSTM_PARAMS),
}

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ===================================================================
# 运行逻辑
# ===================================================================

def run_all(
    model_names: list[str] | None = None,
    csv_path: str | None = None,
    split_ratio: float = SPLIT_RATIO,
    val_ratio: float = VAL_RATIO,
    output_path: str | None = None,
) -> dict:
    """依次训练/评估所选模型，汇总并保存结果。"""

    from src.config import ModelConfig
    from src.compare import ml_baselines as M

    config = ModelConfig

    if csv_path is None:
        csv_path = getattr(config, "DATA_CSV", os.path.join("data", "GOOGL_processed.csv"))

    feature_cols = list(getattr(config, "FEATURE_COLS", []))
    target_col = getattr(config, "TARGET_COL", "Close")

    if model_names is None:
        model_names = ALL_MODEL_NAMES

    all_results: dict[str, dict] = {}

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model '{name}', skipping.")
            continue

        func_name, params = MODEL_REGISTRY[name]
        evaluate_fn = getattr(M, func_name)

        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        result = evaluate_fn(
            csv_path=csv_path,
            feature_cols=feature_cols,
            target_col=target_col,
            split_ratio=split_ratio,
            val_ratio=val_ratio,
            config=config,
            **params,
        )
        all_results[name] = result

    # ---- 打印汇总 ----
    _print_summary(all_results)

    # ---- 保存 JSON ----
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent / "ml_baselines.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_results


def _print_summary(results: dict[str, dict]) -> None:
    """在终端输出对比表格。"""
    header = f"{'Model':<28} {'Train MSE':>14} {'Val MSE':>14} {'Test MSE':>14}"
    line = "-" * len(header)
    print(f"\n{line}")
    print(header)
    print(line)
    for name, res in results.items():
        m = res.get("metrics", {})
        tr = m.get("train", {}).get("mse_raw", float("nan"))
        va = m.get("val", {}).get("mse_raw", float("nan"))
        te = m.get("test", {}).get("mse_raw", float("nan"))
        print(f"{name:<28} {tr:>14.4f} {va:>14.4f} {te:>14.4f}")
    print(line)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ML / CNN baselines for stock prediction comparison.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        choices=ALL_MODEL_NAMES,
        help=f"Models to run (default: all). Choices: {ALL_MODEL_NAMES}",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to processed CSV")
    parser.add_argument("--window", type=int, default=None, help="Override window_size for all models")
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    # 如果通过 CLI 覆盖了窗口大小，更新所有模型参数
    if args.window is not None:
        for _, (_, params) in MODEL_REGISTRY.items():
            params["window_size"] = args.window

    run_all(
        model_names=args.models,
        csv_path=args.data,
        split_ratio=args.split_ratio,
        val_ratio=args.val_ratio,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
