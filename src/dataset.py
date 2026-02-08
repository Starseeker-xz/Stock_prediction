import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, csv_path, config, mode='train', split_ratio=0.7, val_ratio=0.15, device='cpu'):
        """
        Args:
            csv_path: 数据文件路径
            config: 配置对象
            mode: 'train', 'val' or 'test'
            split_ratio: 训练集比例 (default 0.7)
            val_ratio: 验证集比例 (default 0.15)
            device: torch device (default 'cpu')
        """
        self.config = config
        self.device = device
        
        # 1. 读取数据（已预处理的 CSV，包含对数收益）
        # 预处理后列顺序: Date, Open, High, Low, Close, Volume, Open_logret, High_logret, Low_logret, Close_logret
        df = pd.read_csv(csv_path)
        
        # 使用配置化的特征与目标
        feature_cols_long = list(getattr(self.config, 'FEATURE_COLS_LONG', self.config.FEATURE_COLS))
        feature_cols_short_medium = list(getattr(self.config, 'FEATURE_COLS_SHORT_MEDIUM', self.config.FEATURE_COLS))

        # 识别聚合方式: _logret 结尾用 Sum, 其他用 Mean (针对长序列窗口聚合)
        self.sum_feature_indices = [i for i, col in enumerate(feature_cols_long) if col.endswith('_logret')]
        self.mean_feature_indices = [i for i, col in enumerate(feature_cols_long) if not col.endswith('_logret')]

        target_col = self.config.TARGET_COL
        feature_norm_mask_long = list(getattr(self.config, 'FEATURE_NORMALIZE_MASK_LONG', [True] * len(feature_cols_long)))
        feature_norm_mask_short_medium = list(
            getattr(self.config, 'FEATURE_NORMALIZE_MASK_SHORT_MEDIUM', [True] * len(feature_cols_short_medium))
        )
        target_normalize = bool(getattr(self.config, 'TARGET_NORMALIZE', True))

        if len(feature_norm_mask_long) != len(feature_cols_long):
            raise ValueError(
                f"FEATURE_NORMALIZE_MASK_LONG length ({len(feature_norm_mask_long)}) must match FEATURE_COLS_LONG length ({len(feature_cols_long)})"
            )
        if len(feature_norm_mask_short_medium) != len(feature_cols_short_medium):
            raise ValueError(
                f"FEATURE_NORMALIZE_MASK_SHORT_MEDIUM length ({len(feature_norm_mask_short_medium)}) must match FEATURE_COLS_SHORT_MEDIUM length ({len(feature_cols_short_medium)})"
            )

        # 对数收益首行会有 NaN，去除含NaN的行
        required_cols = list(dict.fromkeys(feature_cols_long + feature_cols_short_medium))
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        data_features_long = df[feature_cols_long].values.astype(np.float32)
        data_features_short_medium = df[feature_cols_short_medium].values.astype(np.float32)
        # 目标为配置列
        target_series = df[target_col].values.astype(np.float32)
        
        # 2. 数据标准化（支持按列选择性归一化）
        # 特征：对 mask=True 的列做 StandardScaler，其余列保持原值
        # 目标：由 TARGET_NORMALIZE 控制
        self.feature_scaler_long = StandardScaler()
        self.feature_scaler_short_medium = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # self.feature_scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()
        
        total_samples = len(data_features_long)
        train_size = int(total_samples * split_ratio)
        val_size = int(total_samples * val_ratio)
        val_end = train_size + val_size

        feature_norm_idx_long = np.array([i for i, flag in enumerate(feature_norm_mask_long) if flag], dtype=int)
        if feature_norm_idx_long.size > 0:
            self.feature_scaler_long.fit(data_features_long[:train_size, feature_norm_idx_long])
            features_long_scaled = data_features_long.copy()
            features_long_scaled[:, feature_norm_idx_long] = self.feature_scaler_long.transform(
                data_features_long[:, feature_norm_idx_long]
            )
        else:
            features_long_scaled = data_features_long

        feature_norm_idx_short_medium = np.array(
            [i for i, flag in enumerate(feature_norm_mask_short_medium) if flag], dtype=int
        )
        if feature_norm_idx_short_medium.size > 0:
            self.feature_scaler_short_medium.fit(data_features_short_medium[:train_size, feature_norm_idx_short_medium])
            features_short_medium_scaled = data_features_short_medium.copy()
            features_short_medium_scaled[:, feature_norm_idx_short_medium] = self.feature_scaler_short_medium.transform(
                data_features_short_medium[:, feature_norm_idx_short_medium]
            )
        else:
            features_short_medium_scaled = data_features_short_medium

        if target_normalize:
            self.target_scaler.fit(target_series[:train_size].reshape(-1, 1))
            target_scaled = self.target_scaler.transform(target_series.reshape(-1, 1)).astype(np.float32)
        else:
            target_scaled = target_series.reshape(-1, 1).astype(np.float32)
        
        # 3. 存储全量数据，通过 indices 区分
        self.features_long = features_long_scaled
        self.features_short_medium = features_short_medium_scaled
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
        
        total_len = len(self.features_long)
        
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
            
        # --- Pre-calculate all data and move to Device ---
        # 优化：数据处理完一次性放进gpu，此后训练过程除了把loss传回来完全不和cpu交互
        print(f"[{mode}] Pre-processing {len(self.valid_indices)} samples to {self.device}...")

        x_long_list = []
        x_medium_list = []
        x_short_list = []
        target_list = []
        prev_target_list = []
        
        # 提取配置，减少循环内的查找开销
        seq_len_long = self.config.SEQ_LEN_LONG
        seq_len_medium = self.config.SEQ_LEN_MEDIUM
        seq_len_short = self.config.SEQ_LEN_SHORT
        long_num_windows = self.config.LONG_NUM_WINDOWS
        long_window_size = self.config.LONG_WINDOW_SIZE
        
        for idx in self.valid_indices:
            # Long sequence
            start_long = idx - seq_len_long + 1
            raw_long = self.features_long[start_long : idx + 1]
            
            # Reshape & Sum/Mean
            x_long_windowed = raw_long.reshape(long_num_windows, long_window_size, -1)
            x_long = np.zeros((long_num_windows, raw_long.shape[1]), dtype=np.float32)
            
            if self.sum_feature_indices:
                x_long[:, self.sum_feature_indices] = x_long_windowed[:, :, self.sum_feature_indices].sum(axis=1)
            if self.mean_feature_indices:
                x_long[:, self.mean_feature_indices] = x_long_windowed[:, :, self.mean_feature_indices].mean(axis=1)
            
            x_long_list.append(x_long)
            
            # Medium sequence
            start_medium = idx - seq_len_medium + 1
            x_medium_list.append(self.features_short_medium[start_medium : idx + 1])
            
            # Short sequence
            start_short = idx - seq_len_short + 1
            x_short_list.append(self.features_short_medium[start_short : idx + 1])
            
            # Target (y_{t+1}) and previous target (y_t)
            target_list.append(self.target[idx + 1, 0])
            prev_target_list.append(self.target[idx, 0])
            
        print(f"[{mode}] Stacking and moving to {self.device}...")
        
        if len(self.valid_indices) > 0:
            self.x_long_tensor = torch.tensor(np.stack(x_long_list), dtype=torch.float32).to(self.device).detach()
            self.x_medium_tensor = torch.tensor(np.stack(x_medium_list), dtype=torch.float32).to(self.device).detach()
            self.x_short_tensor = torch.tensor(np.stack(x_short_list), dtype=torch.float32).to(self.device).detach()
            self.target_tensor = torch.tensor(np.stack(target_list), dtype=torch.float32).reshape(-1, 1).to(self.device).detach()
            self.prev_target_tensor = torch.tensor(np.stack(prev_target_list), dtype=torch.float32).reshape(-1, 1).to(self.device).detach()
        else:
             self.x_long_tensor = torch.empty(0).to(self.device)
             self.x_medium_tensor = torch.empty(0).to(self.device)
             self.x_short_tensor = torch.empty(0).to(self.device)
             self.target_tensor = torch.empty(0).to(self.device)
             self.prev_target_tensor = torch.empty(0).to(self.device)

        # Cleanup CPU memory
        # del self.features_long
        # del self.features_short_medium
        # del self.target
            
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        # Directly return GPU tensors
        # Note: No checking for valid_indices here, we assume index corresponds to the dense tensors we built
        return (
            self.x_long_tensor[index],
            self.x_medium_tensor[index],
            self.x_short_tensor[index],
            self.target_tensor[index],
            self.prev_target_tensor[index]
        )
