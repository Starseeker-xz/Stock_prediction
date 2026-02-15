"""ML / CNN baseline models for stock prediction comparison.

Implements:
  - Linear Regression
  - Polynomial Regression
  - Random Forest
  - Support Vector Machine (SVR)
  - CNN (1D Convolution, PyTorch)

All models use a sliding‑window approach:
  Input :  past ``window_size`` days of ``FEATURE_COLS`` → flatten (ML) or 2‑D (CNN)
  Target:  next‑day ``TARGET_COL`` (raw scale)

Evaluation is done on the same train / val / test splits produced by ``StockDataset``
so that MSE numbers are directly comparable with the hierarchical model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _ensure_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_ensure_project_root()

from src.compare._common import build_results_skeleton, load_series_raw, mse  # noqa: E402


# ===================================================================
# Data preparation (shared by all models)
# ===================================================================

def prepare_data(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Build (X, y) arrays using a sliding window, split into train / val / test.

    Returns
    -------
    dict with keys ``X_train, y_train, X_val, y_val, X_test, y_test``
      (y 已归一化),  ``y_train_raw, y_val_raw, y_test_raw`` (原始尺度),
      ``feature_scaler, target_scaler, idx_train, idx_val, idx_test``.
    """
    from sklearn.preprocessing import StandardScaler

    df, y_raw = load_series_raw(csv_path, feature_cols=feature_cols, target_col=target_col)
    features = df[feature_cols].to_numpy(dtype=np.float64)

    total = len(features)
    train_end = int(total * split_ratio)
    val_end = train_end + int(total * val_ratio)

    # --- Build sliding‑window samples ---
    # For index t (last day of input window), target is y[t+1]
    X_all, y_all, idx_all = [], [], []
    for t in range(window_size - 1, total - 1):
        window = features[t - window_size + 1: t + 1]  # shape (window_size, n_features)
        X_all.append(window)
        y_all.append(y_raw[t + 1])
        idx_all.append(t)

    X_all = np.array(X_all, dtype=np.float64)  # (N, window_size, n_features)
    y_all = np.array(y_all, dtype=np.float64)
    idx_all = np.array(idx_all, dtype=np.int64)

    # --- Split by the *last day index* of each window ---
    mask_train = idx_all < train_end
    mask_val = (idx_all >= train_end) & (idx_all < val_end)
    mask_test = idx_all >= val_end

    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_val, y_val = X_all[mask_val], y_all[mask_val]
    X_test, y_test = X_all[mask_test], y_all[mask_test]

    # --- Standardise features (fit on train only) ---
    n_feat = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def _scale(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(orig_shape)

    X_train = _scale(X_train)
    X_val = _scale(X_val)
    X_test = _scale(X_test)

    # --- Standardise target (fit on train only) ---
    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, 1))

    y_train_raw, y_val_raw, y_test_raw = y_train.copy(), y_val.copy(), y_test.copy()
    y_train = target_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val   = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test  = target_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        # 原始尺度的 y，用于反归一化后计算 MSE
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "feature_scaler": scaler,
        "target_scaler": target_scaler,
        "idx_train": idx_all[mask_train],
        "idx_val": idx_all[mask_val],
        "idx_test": idx_all[mask_test],
    }


def _flatten(X: np.ndarray) -> np.ndarray:
    """(N, window, feat) → (N, window*feat)"""
    return X.reshape(X.shape[0], -1)


def _inverse_target(pred_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """将归一化空间的预测值反变换回原始尺度。"""
    return target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(-1)


# ===================================================================
# 1. Linear Regression
# ===================================================================

def evaluate_linear_regression(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
) -> dict:
    from sklearn.linear_model import LinearRegression

    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)
    model = LinearRegression()
    model.fit(_flatten(data["X_train"]), data["y_train"])

    results = build_results_skeleton(
        "linear_regression",
        f"Linear Regression baseline (window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {"window_size": window_size}

    for split in ("train", "val", "test"):
        pred_scaled = model.predict(_flatten(data[f"X_{split}"]))
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results


# ===================================================================
# 2. Polynomial Regression
# ===================================================================

def evaluate_polynomial_regression(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    degree: int = 2,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)

    X_tr_flat = _flatten(data["X_train"])

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr_flat)

    # Use Ridge to avoid ill‑conditioning with high‑dimensional polynomial features
    model = Ridge(alpha=1.0)
    model.fit(X_tr_poly, data["y_train"])

    results = build_results_skeleton(
        "polynomial_regression",
        f"Polynomial Regression (degree={degree}, window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {"window_size": window_size, "degree": degree}

    for split in ("train", "val", "test"):
        X_poly = poly.transform(_flatten(data[f"X_{split}"]))
        pred_scaled = model.predict(X_poly)
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results


# ===================================================================
# 3. Random Forest
# ===================================================================

def evaluate_random_forest(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    n_estimators: int = 200,
    max_depth: int | None = 12,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
    seed: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(_flatten(data["X_train"]), data["y_train"])

    results = build_results_skeleton(
        "random_forest",
        f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {
        "window_size": window_size,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "seed": seed,
    }

    for split in ("train", "val", "test"):
        pred_scaled = model.predict(_flatten(data[f"X_{split}"]))
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results


# ===================================================================
# 4. Support Vector Machine (SVR)
# ===================================================================

def evaluate_svm(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    kernel: str = "rbf",
    C: float = 10.0,
    epsilon: float = 0.1,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
) -> dict:
    from sklearn.svm import SVR

    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(_flatten(data["X_train"]), data["y_train"])

    results = build_results_skeleton(
        "svm",
        f"SVR (kernel={kernel}, C={C}, eps={epsilon}, window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {
        "window_size": window_size,
        "kernel": kernel,
        "C": C,
        "epsilon": epsilon,
    }

    for split in ("train", "val", "test"):
        pred_scaled = model.predict(_flatten(data[f"X_{split}"]))
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results


# ===================================================================
# 5. CNN (1D Convolution)
# ===================================================================

class _CNN1D_Model:
    """Lightweight 1D‑CNN regressor built with PyTorch.

    Architecture
    ------------
    Input shape : (batch, window_size, n_features)
    → Conv1d layers with ReLU & BatchNorm
    → Global average pooling → FC → 1
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 20,
        seed: int = 42,
        device: str | None = None,
    ):
        import torch
        import torch.nn as nn

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if channels is None:
            channels = [32, 64, 64]

        layers: list[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            pad = (kernel_size - 1) // 2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad))
            # layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers).to(self.device)
        self.head = nn.Linear(in_ch, 1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=lr, weight_decay=1e-4,
        )

    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray, y: np.ndarray | None = None):
        import torch
        # X: (N, window, feat) → Conv1d expects (N, feat, window)
        Xt = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            return Xt, yt
        return Xt

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        Xt, yt = self._to_tensor(X_train, y_train)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv, yv = self._to_tensor(X_val, y_val)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.encoder.train()
            self.head.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                h = self.encoder(xb)                    # (B, C, W)
                h = h.mean(dim=2)                       # global avg pool → (B, C)
                out = self.head(h)                       # (B, 1)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.head.parameters()), 1.0
                )
                self.optimizer.step()
                epoch_loss += loss.item()

            if has_val:
                val_loss = self._eval_loss(Xv, yv)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        "encoder": {k: v.clone() for k, v in self.encoder.state_dict().items()},
                        "head": {k: v.clone() for k, v in self.head.state_dict().items()},
                    }
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.head.load_state_dict(best_state["head"])

    def _eval_loss(self, Xt, yt) -> float:
        import torch
        self.encoder.eval()
        self.head.eval()
        with torch.no_grad():
            h = self.encoder(Xt).mean(dim=2)
            out = self.head(h)
            return float(self.criterion(out, yt).item())

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.encoder.eval()
        self.head.eval()
        Xt = self._to_tensor(X)
        with torch.no_grad():
            h = self.encoder(Xt).mean(dim=2)
            out = self.head(h)
        return out.cpu().numpy().reshape(-1)


def evaluate_cnn1d(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    channels: list[int] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
    seed: int = 42,
) -> dict:
    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)
    n_features = data["X_train"].shape[-1]

    if channels is None:
        channels = [32, 64, 64]

    model = _CNN1D_Model(
        n_features=n_features,
        window_size=window_size,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
    model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    results = build_results_skeleton(
        "cnn_1d",
        f"1D‑CNN (channels={channels}, kernel={kernel_size}, window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {
        "window_size": window_size,
        "channels": channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
    }

    for split in ("train", "val", "test"):
        pred_scaled = model.predict(data[f"X_{split}"])
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results


# ===================================================================
# 6. Simple LSTM
# ===================================================================

class _SimpleLSTM_Model:
    """Single‐layer LSTM regressor built with PyTorch.

    Architecture
    ------------
    Input shape : (batch, window_size, n_features)
    → LSTM(hidden_size) → take last hidden state → FC → 1
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 20,
        seed: int = 42,
        device: str | None = None,
    ):
        import torch
        import torch.nn as nn

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        ).to(self.device)

        self.dropout = nn.Dropout(dropout).to(self.device)
        self.head = nn.Linear(hidden_size, 1).to(self.device)
        self.criterion = nn.MSELoss()

        params = list(self.lstm.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray, y: np.ndarray | None = None):
        import torch
        # X: (N, window, feat) — already in LSTM’s expected layout
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
            return Xt, yt
        return Xt

    def _forward(self, Xt):
        # output: (B, T, H)  → take last time step
        output, _ = self.lstm(Xt)
        h_last = output[:, -1, :]  # (B, H)
        h_last = self.dropout(h_last)
        return self.head(h_last)    # (B, 1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        Xt, yt = self._to_tensor(X_train, y_train)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv, yv = self._to_tensor(X_val, y_val)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.lstm.train()
            self.head.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                out = self._forward(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.lstm.parameters()) + list(self.head.parameters()), 1.0
                )
                self.optimizer.step()

            if has_val:
                val_loss = self._eval_loss(Xv, yv)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        "lstm": {k: v.clone() for k, v in self.lstm.state_dict().items()},
                        "head": {k: v.clone() for k, v in self.head.state_dict().items()},
                    }
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.lstm.load_state_dict(best_state["lstm"])
            self.head.load_state_dict(best_state["head"])

    def _eval_loss(self, Xt, yt) -> float:
        import torch
        self.lstm.eval()
        self.head.eval()
        with torch.no_grad():
            out = self._forward(Xt)
            return float(self.criterion(out, yt).item())

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.lstm.eval()
        self.head.eval()
        Xt = self._to_tensor(X)
        with torch.no_grad():
            out = self._forward(Xt)
        return out.cpu().numpy().reshape(-1)


def evaluate_lstm(
    csv_path: str,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 5,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 20,
    split_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Any = None,
    seed: int = 42,
) -> dict:
    data = prepare_data(csv_path, feature_cols, target_col, window_size, split_ratio, val_ratio)
    n_features = data["X_train"].shape[-1]

    model = _SimpleLSTM_Model(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
    model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    results = build_results_skeleton(
        "lstm",
        f"Simple LSTM (hidden={hidden_size}, layers={num_layers}, window={window_size})",
        csv_path, split_ratio, val_ratio, config,
    )
    results["params"] = {
        "window_size": window_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
    }

    for split in ("train", "val", "test"):
        pred_scaled = model.predict(data[f"X_{split}"])
        pred_raw = _inverse_target(pred_scaled, data["target_scaler"])
        results["metrics"][split] = {
            "mse_raw": mse(pred_raw, data[f"y_{split}_raw"]),
            "n": int(len(pred_raw)),
        }
    return results
