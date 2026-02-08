import numpy as np
import pandas as pd

# 修改这里为你的 CSV 路径（要求前两列分别是 y_true, y_pred）
CSV_PATH = r"checkpoints\\<timestamp>\\pred_test.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least 2 columns, got {df.shape[1]}")

    y_true = df.iloc[:, 0].to_numpy(dtype=np.float64).reshape(-1)
    y_pred = df.iloc[:, 1].to_numpy(dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    mse = float(np.mean((y_pred - y_true) ** 2))
    print(f"CSV: {CSV_PATH}")
    print(f"MSE (first two cols): {mse:.6f}")


if __name__ == "__main__":
    main()
