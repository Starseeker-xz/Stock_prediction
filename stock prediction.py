

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =======================
# Config
# =======================
CSV_PATH = 'data\GOOGL_processed.csv'

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET   = 'Close'

SHORT_WIN = 3
MID_WIN   = 10
LONG_WIN  = 15
# type = 'short'
# type = 'short+mid'
type = 'short+mid+long'

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================
# Model
# =======================
class ShortTermCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class MidTermRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


class LongTermBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class MultiScalePredictor(nn.Module):
    def __init__(self, feature_dim, type):
        super().__init__()
        self.type = type
        self.short_net = ShortTermCNN(feature_dim, 64)
        self.mid_net   = MidTermRNN(feature_dim, 64)
        self.long_net  = LongTermBiLSTM(feature_dim, 64)

        if type == 'short':
            fusion_dim = 64 
        elif type == 'short+mid':
            fusion_dim = 64 + 64
        elif type == 'short+long':
            fusion_dim = 64 + 2 * 64
        elif type == 'mid+long':
            fusion_dim = 64 + 2 * 64
        elif type == 'short+mid+long':
            fusion_dim = 64 + 64 + 2 * 64
        else:
            raise ValueError(f"Invalid type: {type}")

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, xs, xm, xl):
        
        if self.type == 'short':
            x = self.short_net(xs)
            return self.fc(x)
        elif self.type == 'short+mid':
            fs = self.short_net(xs)
            fm = self.mid_net(xm)
            return self.fc(torch.cat([fs, fm], dim=1))
        elif self.type == 'short+long':
            fs = self.short_net(xs)
            fl = self.long_net(xl)
            return self.fc(torch.cat([fs, fl], dim=1))
        elif self.type == 'mid+long':
            fm = self.mid_net(xm)
            fl = self.long_net(xl)
            return self.fc(torch.cat([fm, fl], dim=1))
        elif self.type == 'short+mid+long':
            fs = self.short_net(xs)
            fm = self.mid_net(xm)
            fl = self.long_net(xl)
            return self.fc(torch.cat([fs, fm, fl], dim=1))
        else:
            raise ValueError(f"Invalid type: {self.type}")


# =======================
# Dataset
# =======================
class StockDataset(Dataset):
    def __init__(self, xs, xm, xl, y):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.xm = torch.tensor(xm, dtype=torch.float32)
        self.xl = torch.tensor(xl, dtype=torch.float32)
        self.y  = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.xs[idx], self.xm[idx], self.xl[idx], self.y[idx]


# =======================
# Utils
# =======================
def build_multiscale_dataset(X, y, sw, mw, lw):
    xs, xm, xl, ys = [], [], [], []
    max_w = max(sw, mw, lw)

    for i in range(max_w, len(X)):
        xs.append(X[i-sw:i])
        xm.append(X[i-mw:i])
        xl.append(X[i-lw:i])
        ys.append(y[i])

    return np.array(xs), np.array(xm), np.array(xl), np.array(ys)


def standardize(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (
        (train-mean)/std,
        (val-mean)/std,
        (test-mean)/std
    )


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = 0
    for xs, xm, xl, y in loader:
        xs, xm, xl, y = xs.to(DEVICE), xm.to(DEVICE), xl.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xs, xm, xl), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for xs, xm, xl, y in loader:
            xs, xm, xl, y = xs.to(DEVICE), xm.to(DEVICE), xl.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(xs, xm, xl), y)
            loss_sum += loss.item()
    return loss_sum / len(loader)


# =======================
# Main
# =======================
def main():
    df = pd.read_csv(CSV_PATH)

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    n = len(X)
    tr, va = int(0.7*n), int(0.85*n)

    X_tr, X_va, X_te = X[:tr], X[tr:va], X[va:]
    y_tr, y_va, y_te = y[:tr], y[tr:va], y[va:]

    X_tr, X_va, X_te = standardize(X_tr, X_va, X_te)

    Xs_tr, Xm_tr, Xl_tr, y_tr = build_multiscale_dataset(
        X_tr, y_tr, SHORT_WIN, MID_WIN, LONG_WIN
    )
    Xs_va, Xm_va, Xl_va, y_va = build_multiscale_dataset(
        X_va, y_va, SHORT_WIN, MID_WIN, LONG_WIN
    )
    Xs_te, Xm_te, Xl_te, y_te = build_multiscale_dataset(
        X_te, y_te, SHORT_WIN, MID_WIN, LONG_WIN
    )

    train_loader = DataLoader(
        StockDataset(Xs_tr, Xm_tr, Xl_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        StockDataset(Xs_va, Xm_va, Xl_va, y_va),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        StockDataset(Xs_te, Xm_te, Xl_te, y_te),
        batch_size=BATCH_SIZE
    )

    model = MultiScalePredictor(len(FEATURES),type = type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        va_loss = eval_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train MSE: {tr_loss:.4f} | Val MSE: {va_loss:.4f}")

    # Test
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xs, xm, xl, y in test_loader:
            pred = model(xs.to(DEVICE), xm.to(DEVICE), xl.to(DEVICE))
            preds.append(pred.cpu().numpy())
            gts.append(y.numpy())

    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)

    print("Test RMSE:", np.sqrt(mean_squared_error(gts, preds)))
    print("Test MAE :", mean_absolute_error(gts, preds))


if __name__ == '__main__':
    main()
