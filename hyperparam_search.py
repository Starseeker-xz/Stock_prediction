"""
超参数搜索脚本
==============
两阶段：
  阶段1: 在 full 模式下搜索最佳超参数组合
  阶段2: 用找到的最佳超参数，分别跑 short_only / short_medium / full 三种模式做对比

结果保存在 checkpoints/hp_search_<timestamp>/ 下。
用法:
    python hyperparam_search.py
"""

import os
import sys
import json
import itertools
import datetime
import traceback

from src.config import ModelConfig
from train import train


# ========== 超参数搜索空间 ==========

SEARCH_SPACE = {
    # 序列长度
    "SEQ_LEN_SHORT": [5, 10],
    "SEQ_LEN_MEDIUM": [10, 21],
    "LONG_NUM_WINDOWS": [21, 63],
    "LONG_WINDOW_SIZE": [1],

    # 模型容量
    "TCN_CHANNELS": [
        [32, 32, 64],
        [32, 64, 64, 64],
    ],
    "HIDDEN_SIZE_MEDIUM": [32, 64],
    "HIDDEN_SIZE_LONG": [32],
    "D_MODEL": [64],

    # 融合
    "FUSION_METHOD": ["softmax_short_mlp"],
    "BRANCH_LAYER_NORM_MODE": ["all", "aux_only"],
    "GATE_INIT_BIAS": [-1.0],

    # 正则化
    "DROPOUT": [0.1, 0.2],
    "WEIGHT_DECAY": [0.0, 1e-4],

    # 学习率
    "LEARNING_RATE": [1e-4, 3e-4],
}

# 固定不搜索的参数（覆盖 config 里的值）
FIXED_PARAMS = {
    "MODEL_MODE": "full",
    "INSTANCE_NORM": True,
    "BRANCH_LAYER_NORM": True,
    "GATE_HIDDEN_SIZE": 32,
    "GATE_DETACH_AUX": False,
    "PATIENCE": 40,
    "BATCH_SIZE": 32,
    "EPOCHS": 1000,
    "SEED": 42,
    "PREDICT_RESIDUAL": False,
    "TARGET_NORMALIZE": True,
}

# 阶段2对比的模式
COMPARE_MODES = ["short_only", "short_medium", "full"]

# 搜索结果排序指标
METRIC_KEY = "best_val_loss"  # 按 val loss 排序


# ========== 工具函数 ==========

def _build_param_grid(search_space: dict) -> list[dict]:
    """展开搜索空间为参数组合列表。"""
    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _apply_params(params: dict):
    """将参数 dict 写入 ModelConfig（类变量）。"""
    for key, val in params.items():
        setattr(ModelConfig, key, val)
    # 重新计算依赖字段
    ModelConfig.SEQ_LEN_LONG = ModelConfig.LONG_WINDOW_SIZE * ModelConfig.LONG_NUM_WINDOWS
    ModelConfig.INPUT_SIZE_LONG = len(ModelConfig.FEATURE_COLS_LONG)
    ModelConfig.INPUT_SIZE_SHORT_MEDIUM = len(ModelConfig.FEATURE_COLS_SHORT_MEDIUM)
    ModelConfig.FEATURE_NORMALIZE_MASK = ModelConfig.FEATURE_NORMALIZE_MASK_SHORT_MEDIUM


def _params_to_label(params: dict) -> str:
    """生成简短的文件夹名。"""
    parts = []
    parts.append(f"s{params.get('SEQ_LEN_SHORT', '?')}")
    parts.append(f"m{params.get('SEQ_LEN_MEDIUM', '?')}")
    lw = params.get('LONG_WINDOW_SIZE', 1)
    ln = params.get('LONG_NUM_WINDOWS', '?')
    parts.append(f"l{lw}x{ln}")
    tcn = params.get('TCN_CHANNELS', [])
    parts.append(f"tcn{'_'.join(str(c) for c in tcn)}")
    parts.append(f"hm{params.get('HIDDEN_SIZE_MEDIUM', '?')}")
    parts.append(f"hl{params.get('HIDDEN_SIZE_LONG', '?')}")
    parts.append(f"bn{params.get('BRANCH_LAYER_NORM_MODE', '?')}")
    parts.append(f"gb{params.get('GATE_INIT_BIAS', '?')}")
    parts.append(f"do{params.get('DROPOUT', '?')}")
    parts.append(f"wd{params.get('WEIGHT_DECAY', '?')}")
    parts.append(f"lr{params.get('LEARNING_RATE', '?')}")
    return "_".join(str(p) for p in parts)


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ========== 主流程 ==========

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("checkpoints", f"hp_search_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    # 保存搜索配置
    _dump_json(os.path.join(root_dir, "search_config.json"), {
        "search_space": {k: str(v) for k, v in SEARCH_SPACE.items()},
        "fixed_params": {k: str(v) for k, v in FIXED_PARAMS.items()},
        "compare_modes": COMPARE_MODES,
    })

    param_grid = _build_param_grid(SEARCH_SPACE)
    total = len(param_grid)
    print(f"{'='*80}")
    print(f"Hyperparameter search: {total} combinations")
    print(f"Results will be saved to: {root_dir}")
    print(f"{'='*80}\n")

    # ────────────────────────────────────────────
    # 阶段 1: full 模式超参数搜索
    # ────────────────────────────────────────────
    phase1_dir = os.path.join(root_dir, "phase1_full_search")
    os.makedirs(phase1_dir, exist_ok=True)

    results = []

    for i, params in enumerate(param_grid):
        run_label = _params_to_label(params)
        run_dir = os.path.join(phase1_dir, f"{i:04d}_{run_label}")
        print(f"\n{'='*80}")
        print(f"[Phase 1] Run {i+1}/{total}: {run_label}")
        print(f"{'='*80}")

        # 合并 fixed + search 参数
        all_params = {**FIXED_PARAMS, **params}
        _apply_params(all_params)

        try:
            result = train(save_dir=run_dir)
            entry = {
                "run_id": i,
                "label": run_label,
                "params": {k: v if not isinstance(v, list) else v for k, v in all_params.items()},
                "best_val_loss": result["best_val_loss"],
                "mse_raw": result["mse_raw"],
                "exp_dir": result["exp_dir"],
                "status": "ok",
            }
        except Exception as e:
            traceback.print_exc()
            entry = {
                "run_id": i,
                "label": run_label,
                "params": {k: v if not isinstance(v, list) else v for k, v in all_params.items()},
                "best_val_loss": float("inf"),
                "mse_raw": None,
                "exp_dir": run_dir,
                "status": f"error: {e}",
            }

        results.append(entry)

        # 每跑完一次就写结果，防止中途挂掉
        _dump_json(os.path.join(phase1_dir, "results.json"), results)
        print(f"  => val_loss={entry['best_val_loss']:.6f}, "
              f"test_mse={entry['mse_raw']['test'] if entry['mse_raw'] else 'N/A'}")

    # 排序找最佳
    results_ok = [r for r in results if r["status"] == "ok"]
    results_ok.sort(key=lambda r: r[METRIC_KEY])
    _dump_json(os.path.join(phase1_dir, "results_sorted.json"), results_ok)

    if not results_ok:
        print("\n[ERROR] All runs failed. Aborting.")
        return

    best = results_ok[0]
    best_params = best["params"]
    print(f"\n{'='*80}")
    print(f"[Phase 1] Best run: {best['label']}")
    print(f"  val_loss = {best['best_val_loss']:.6f}")
    print(f"  test_mse = {best['mse_raw']['test']:.6f}")
    print(f"  params   = {json.dumps(best_params, indent=4, default=str)}")
    print(f"{'='*80}\n")

    _dump_json(os.path.join(root_dir, "best_params.json"), best_params)

    # ────────────────────────────────────────────
    # 阶段 2: 用最佳参数跑三种模式做对比
    # ────────────────────────────────────────────
    phase2_dir = os.path.join(root_dir, "phase2_mode_compare")
    os.makedirs(phase2_dir, exist_ok=True)

    compare_results = []
    seeds = [42, 123, 777]  # 多种子取均值

    for mode in COMPARE_MODES:
        mode_results_per_seed = []
        for seed in seeds:
            mode_dir = os.path.join(phase2_dir, f"{mode}_seed{seed}")
            print(f"\n{'='*80}")
            print(f"[Phase 2] Mode={mode}, Seed={seed}")
            print(f"{'='*80}")

            run_params = {**best_params, "MODEL_MODE": mode, "SEED": seed}
            _apply_params(run_params)

            try:
                result = train(save_dir=mode_dir)
                seed_entry = {
                    "mode": mode,
                    "seed": seed,
                    "best_val_loss": result["best_val_loss"],
                    "mse_raw": result["mse_raw"],
                    "exp_dir": result["exp_dir"],
                    "status": "ok",
                }
            except Exception as e:
                traceback.print_exc()
                seed_entry = {
                    "mode": mode,
                    "seed": seed,
                    "best_val_loss": float("inf"),
                    "mse_raw": None,
                    "exp_dir": mode_dir,
                    "status": f"error: {e}",
                }
            mode_results_per_seed.append(seed_entry)

        # 计算均值
        ok_seeds = [r for r in mode_results_per_seed if r["status"] == "ok"]
        if ok_seeds:
            avg_val = sum(r["best_val_loss"] for r in ok_seeds) / len(ok_seeds)
            avg_test = sum(r["mse_raw"]["test"] for r in ok_seeds) / len(ok_seeds)
        else:
            avg_val = float("inf")
            avg_test = float("inf")

        compare_entry = {
            "mode": mode,
            "seeds": seeds,
            "per_seed": mode_results_per_seed,
            "avg_val_loss": avg_val,
            "avg_test_mse": avg_test,
        }
        compare_results.append(compare_entry)
        _dump_json(os.path.join(phase2_dir, "compare_results.json"), compare_results)

    # ────────────────────────────────────────────
    # 最终汇总
    # ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL COMPARISON (avg over seeds)")
    print(f"{'='*80}")
    print(f"{'Mode':<16} {'Avg Val Loss':>14} {'Avg Test MSE':>14}")
    print("-" * 46)
    for cr in compare_results:
        print(f"{cr['mode']:<16} {cr['avg_val_loss']:>14.6f} {cr['avg_test_mse']:>14.6f}")
    print(f"{'='*80}")

    summary = {
        "best_full_params": best_params,
        "comparison": [
            {"mode": cr["mode"], "avg_val_loss": cr["avg_val_loss"], "avg_test_mse": cr["avg_test_mse"]}
            for cr in compare_results
        ],
    }
    _dump_json(os.path.join(root_dir, "summary.json"), summary)
    print(f"\nAll results saved to: {root_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
