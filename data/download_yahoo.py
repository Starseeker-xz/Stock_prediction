import argparse
import os

import numpy as np
import pandas as pd
import yfinance as yf


def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})

    expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f'下载结果缺少列: {missing}. 实际列: {list(df.columns)}')

    out = df[expected].copy()
    out['Date'] = pd.to_datetime(out['Date'])
    out = out.sort_values('Date').reset_index(drop=True)
    return out


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing using EMA
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def preprocess_ohlcv_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """直接将 OHLCV 数据预处理成训练用格式（含 logret 列）。

    输出列：
    - 价格/成交量：Date, Open, High, Low, Close, Volume
    - logret：Open_logret, High_logret, Low_logret, Close_logret
        - RSI：RSI_14
        - 日期周期特征（新增 6 列）：
      WeekOfYear, WeekOfYear_sin, WeekOfYear_cos, DayOfWeek, DayOfWeek_sin, DayOfWeek_cos
    """
    base = ohlcv[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 时间特征：Week of Year / Day of Week，并用 sin/cos 做周期编码
    # ISO week：1-53；DayOfWeek：Monday=0..Sunday=6
    base['Date'] = pd.to_datetime(base['Date'])
    iso = base['Date'].dt.isocalendar()
    week_of_year = iso.week.astype(int)
    day_of_week = base['Date'].dt.dayofweek.astype(int)

    # 为了兼容存在第 53 周的年份，周期长度固定用 53
    week_period = 53.0
    day_period = 7.0

    base['WeekOfYear'] = week_of_year
    base['WeekOfYear_sin'] = np.sin(2.0 * np.pi * ((week_of_year - 1).astype(float)) / week_period)
    base['WeekOfYear_cos'] = np.cos(2.0 * np.pi * ((week_of_year - 1).astype(float)) / week_period)

    base['DayOfWeek'] = day_of_week
    base['DayOfWeek_sin'] = np.sin(2.0 * np.pi * (day_of_week.astype(float)) / day_period)
    base['DayOfWeek_cos'] = np.cos(2.0 * np.pi * (day_of_week.astype(float)) / day_period)

    # RSI (14)
    base['RSI_14'] = _compute_rsi(base['Close'].astype(float), period=14)

    for col in ['Open', 'High', 'Low', 'Close']:
        p = base[col].astype(float).replace({0: np.nan})
        base[f'{col}_logret'] = np.log(p) - np.log(p.shift(1))

    return base


def download_and_preprocess(
    ticker: str,
    out_dir: str = os.path.join('data'),
    interval: str = '1d',
    auto_adjust: bool = False,
) -> tuple[str, str]:
    """下载某个股票的“全部历史”数据并输出 raw + processed 两份 CSV。"""
    out_dir = _ensure_out_dir(out_dir)

    df = yf.download(
        ticker,
        period='max',
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f'未下载到数据: ticker={ticker}, interval={interval}')

    ohlcv = _normalize_ohlcv(df)
    processed = preprocess_ohlcv_df(ohlcv)

    # 尽量与现有命名对齐：{TICKER}_daily.csv / {TICKER}_processed.csv
    raw_name = f'{ticker}_daily.csv' if interval == '1d' else f'{ticker}_{interval}.csv'
    proc_name = (
        f'{ticker}_processed.csv'
        if interval == '1d'
        else f'{ticker}_{interval}_processed.csv'
    )
    raw_path = os.path.join(out_dir, raw_name)
    proc_path = os.path.join(out_dir, proc_name)

    ohlcv.to_csv(raw_path, index=False)
    processed.to_csv(proc_path, index=False)
    return raw_path, proc_path


def main():
    parser = argparse.ArgumentParser(
        description='下载 Yahoo Finance 全量历史数据，并直接预处理为训练用 CSV'
    )
    parser.add_argument(
        'ticker',
        nargs='?',
        default='GOOGL',
        help='股票代码，例如: GOOGL；A股常见后缀如 600519.SS / 000001.SZ',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=os.path.join('data'),
        help='输出目录，默认 data/',
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='周期，默认 1d（需要别的再改，例如 1wk）',
    )
    parser.add_argument(
        '--auto-adjust',
        action='store_true',
        help='使用复权价格(Adjusted)替代原始价格',
    )

    args = parser.parse_args()
    raw_path, proc_path = download_and_preprocess(
        ticker=args.ticker,
        out_dir=args.out_dir,
        interval=args.interval,
        auto_adjust=args.auto_adjust,
    )

    print(f'Raw CSV written: {raw_path}')
    print(f'Processed CSV written: {proc_path}')


if __name__ == '__main__':
    main()
