import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join('data', 'GOOGL_daily.csv')
OUT_PATH = os.path.join('data', 'GOOGL_processed.csv')


def preprocess_raw_csv(input_csv: str = RAW_PATH, output_csv: str = OUT_PATH):
    # 原始文件前三行是元信息，手动赋列名
    # 目标列顺序: Date, Close, High, Low, Open, Volume
    df = pd.read_csv(input_csv, skiprows=3, header=None,
                     names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])

    # 基础清洗与排序
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 选择五个基础参数列，并按常用展示顺序重排: Open, High, Low, Close, Volume
    base = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 计算对数收益率: ln(p_t) - ln(p_{t-1})，仅针对前四个价格列
    for col in ['Open', 'High', 'Low', 'Close']:
        # 避免非正值导致的 log 问题
        p = base[col].astype(float).replace({0: np.nan})
        base[f'{col}_logret'] = np.log(p) - np.log(p.shift(1))

    # 导出为新表
    base.to_csv(output_csv, index=False)
    return output_csv


if __name__ == '__main__':
    out = preprocess_raw_csv()
    print(f'Processed CSV written to: {out}')
