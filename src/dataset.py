import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, csv_path, config, mode='train', split_ratio=0.8, val_ratio=0.1):
        """
        Args:
            csv_path: 数据文件路径
            config: 配置对象
            mode: 'train', 'val' or 'test'
            split_ratio: 训练集比例 (default 0.7)
            val_ratio: 验证集比例 (default 0.15)
        """
        self.config = config
        
        # 1. 读取数据（已预处理的 CSV，包含对数收益）
        # 预处理后列顺序: Date, Open, High, Low, Close, Volume, Open_logret, High_logret, Low_logret, Close_logret
        df = pd.read_csv(csv_path)
        
        # 使用配置化的特征与目标
        feature_cols = list(self.config.FEATURE_COLS)

        # 识别聚合方式: _logret 结尾用 Sum, 其他用 Mean (针对长序列窗口聚合)
        self.sum_feature_indices = [i for i, col in enumerate(feature_cols) if col.endswith('_logret')]
        self.mean_feature_indices = [i for i, col in enumerate(feature_cols) if not col.endswith('_logret')]

        target_col = self.config.TARGET_COL
        feature_norm_mask = list(getattr(self.config, 'FEATURE_NORMALIZE_MASK', [True] * len(feature_cols)))
        target_normalize = bool(getattr(self.config, 'TARGET_NORMALIZE', True))

        if len(feature_norm_mask) != len(feature_cols):
            raise ValueError(
                f"FEATURE_NORMALIZE_MASK length ({len(feature_norm_mask)}) must match FEATURE_COLS length ({len(feature_cols)})"
            )

        # 对数收益首行会有 NaN，去除含NaN的行
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        data_features = df[feature_cols].values.astype(np.float32)
        # 目标为配置列
        target_series = df[target_col].values.astype(np.float32)
        
        # 2. 数据标准化（支持按列选择性归一化）
        # 特征：对 mask=True 的列做 StandardScaler，其余列保持原值
        # 目标：由 TARGET_NORMALIZE 控制
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # self.feature_scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()
        
        total_samples = len(data_features)
        train_size = int(total_samples * split_ratio)
        val_size = int(total_samples * val_ratio)
        val_end = train_size + val_size

        feature_norm_idx = np.array([i for i, flag in enumerate(feature_norm_mask) if flag], dtype=int)
        if feature_norm_idx.size > 0:
            self.feature_scaler.fit(data_features[:train_size, feature_norm_idx])
            features_scaled = data_features.copy()
            features_scaled[:, feature_norm_idx] = self.feature_scaler.transform(data_features[:, feature_norm_idx])
        else:
            features_scaled = data_features

        if target_normalize:
            self.target_scaler.fit(target_series[:train_size].reshape(-1, 1))
            target_scaled = self.target_scaler.transform(target_series.reshape(-1, 1)).astype(np.float32)
        else:
            target_scaled = target_series.reshape(-1, 1).astype(np.float32)
        
        # 3. 存储全量数据，通过 indices 区分
        self.features = features_scaled
        self.target = target_scaled
            
        # 4. 预计算有效索引
        # 我们需要保证 idx - seq_len_long >= 0 且 idx + 1 < len(data)
        # 预测目标：下一日 Close 的对数收益率
        # 假设输入截止到 t, 输出 t+1
        # 所以我们需要 data[t+1] 存在
        
        self.valid_indices = []
        
        # 定义数据集的边界
        # 索引代表输入序列 X 的最后一个时间步 t
        # 预测目标是 t+1
        
        total_len = len(self.features)
        
        if mode == 'train':
            # Train: [config.SEQ_LEN_LONG - 1, train_size)
            start_idx = config.SEQ_LEN_LONG - 1
            end_idx = train_size - 1 
            indices = range(start_idx, train_size)
        elif mode == 'val':
            # Val: [train_size, val_end)
            start_idx = train_size
            indices = range(start_idx, val_end)
        elif mode == 'test':
            # Test: [val_end, total_len - 1)
            # End: total_len - 2 (因为 target 是 idx + 1, 所以 idx 最大是 total_len - 2)
            start_idx = val_end
            indices = range(start_idx, total_len - 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        for i in indices:
            self.valid_indices.append(i)
            
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        # idx 是输入序列最后一个时间步的索引
        idx = self.valid_indices[index]
        
        # 准备输入序列
        # features shape: (total_len, 5)
        
        # Long sequence
        # 使用窗口聚合: (Num_Windows, Window_Size, Feat) -> Sum -> (Num_Windows, Feat)
        start_long = idx - self.config.SEQ_LEN_LONG + 1
        raw_long = self.features[start_long : idx + 1]
        
        # 确保形状正确
        if raw_long.shape[0] != self.config.SEQ_LEN_LONG:
             # 边界情况处理，理论上 valid_indices 保证了不会发生
             raise ValueError(f"Slice length mismatch: {raw_long.shape[0]} vs {self.config.SEQ_LEN_LONG}")

        # Reshape & Sum/Mean
        # reshape: (LONG_NUM_WINDOWS, LONG_WINDOW_SIZE, feature_dim)
        x_long_windowed = raw_long.reshape(
            self.config.LONG_NUM_WINDOWS, 
            self.config.LONG_WINDOW_SIZE, 
            -1
        )
        # 聚合：区分 Sum 和 Mean
        # x_long shape: (LONG_NUM_WINDOWS, feature_dim)
        x_long = np.zeros((self.config.LONG_NUM_WINDOWS, raw_long.shape[1]), dtype=np.float32)
        
        if self.sum_feature_indices:
             # logret 类特征：求和 (Sum Pooling)
            x_long[:, self.sum_feature_indices] = x_long_windowed[:, :, self.sum_feature_indices].sum(axis=1)
            
        if self.mean_feature_indices:
            # 价格/成交量等其他特征：求平均 (Mean Pooling)
            x_long[:, self.mean_feature_indices] = x_long_windowed[:, :, self.mean_feature_indices].mean(axis=1)
        
        # Medium sequence
        start_medium = idx - self.config.SEQ_LEN_MEDIUM + 1
        x_medium = self.features[start_medium : idx + 1]
        
        # Short sequence
        start_short = idx - self.config.SEQ_LEN_SHORT + 1
        x_short = self.features[start_short : idx + 1]
        
        # Target: Next day target (可能未标准化，取决于配置)
        target = self.target[idx + 1, 0]
        
        return (
            torch.tensor(x_long, dtype=torch.float32),
            torch.tensor(x_medium, dtype=torch.float32),
            torch.tensor(x_short, dtype=torch.float32),
            torch.tensor([target], dtype=torch.float32)
        )
