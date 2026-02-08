import numpy as np
import pandas as pd


CSV_PATH = r"checkpoints\20260208_132923\pred_test.csv"


def direction_accuracy_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least 2 columns, got {df.shape[1]}")

    y_true = df.iloc[:, 0].to_numpy(dtype=np.float64).reshape(-1)
    y_pred = df.iloc[:, 1].to_numpy(dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if len(y_true) < 2:
        raise ValueError("Need at least 2 rows to compute direction accuracy.")

    prev_true = y_true[:-1]
    true_next = y_true[1:]
    pred_next = y_pred[1:]

    true_dir = np.sign(true_next - prev_true)
    pred_dir = np.sign(pred_next - prev_true)

    correct = (true_dir == pred_dir)
    acc = float(np.mean(correct))

    return {
        "direction_accuracy": acc,
        "n": int(len(true_dir)),
    }


def main() -> None:
    res = direction_accuracy_from_csv(CSV_PATH)
    print(f"CSV: {CSV_PATH}")
    print(f"Direction Accuracy: {res['direction_accuracy']:.4f} (n={res['n']})")


if __name__ == "__main__":
    main()
