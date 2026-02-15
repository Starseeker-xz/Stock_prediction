
class ModelConfig:
    DATA_CSV = 'data/GOOGL_processed.csv'

    # FEATURE_COLS = ['Open'( 'High', 'Low', 'Close','Open_logret', 'High_logret', 'Low_logret', 'Close_logret', 'Volume']
    # FEATURE_COLS = ['Close', (Volume']
    BASE_FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_logret']

    # 时间特征
    TIME_FEATURES_LONG = ['WeekOfYear_sin', 'WeekOfYear_cos']
    TIME_FEATURES_SHORT_MEDIUM = ['WeekOfYear_sin', 'WeekOfYear_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']

    # 不同头的特征
    FEATURE_COLS_LONG = BASE_FEATURE_COLS
    FEATURE_COLS_SHORT_MEDIUM = BASE_FEATURE_COLS
    # 默认特征（兼容其他模块）
    FEATURE_COLS = FEATURE_COLS_SHORT_MEDIUM
    
    # TARGET_COL = 'Close_logret'
    TARGET_COL = 'Close'
    # 预测残差：输出 = y[t] + f(x<=t)
    PREDICT_RESIDUAL = False
    
    INPUT_SIZE_LONG = len(FEATURE_COLS_LONG)
    INPUT_SIZE_SHORT_MEDIUM = len(FEATURE_COLS_SHORT_MEDIUM)
    OUTPUT_SIZE = 1 

    # 是否归一化
    FEATURE_NORMALIZE_MASK_SHORT_MEDIUM = [True, True, True, True, True, True]
    # + [False] * len(TIME_FEATURES_SHORT_MEDIUM)
    # FEATURE_NORMALIZE_MASK_SHORT_MEDIUM = [True, True, True, True, True]
    # FEATURE_NORMALIZE_MASK_LONG = [True, True, True, True, True]
    FEATURE_NORMALIZE_MASK = FEATURE_NORMALIZE_MASK_SHORT_MEDIUM
    TARGET_NORMALIZE = True

    # 实例归一化 (RevIN): 对每个样本的输入窗口独立做归一化
    # 解决不同时间段价格水平差异导致的分布漂移问题
    # 输出会自动反归一化回全局标准化空间
    INSTANCE_NORM = False
    
    # 长时序列配置：窗口聚合模式
    LONG_WINDOW_SIZE = 5 # 一个窗口聚合多少天
    LONG_NUM_WINDOWS = 13 # 输入多少个窗口
    SEQ_LEN_LONG = LONG_WINDOW_SIZE * LONG_NUM_WINDOWS # 总回看长度
    
    SEQ_LEN_MEDIUM = 10
    SEQ_LEN_SHORT = 5 
    
    # 模型维度
    # 以 short 为骨干的测试模式：
    # 'short_only'（仅短时）, 'short_medium'（短+中）, 'short_long'（短+长）, 'full'（短+中+长）
    # MODEL_MODE = 'full'
    # MODEL_MODE = 'short_only'
    MODEL_MODE = 'short_medium'
    # MODEL_MODE = 'short_long'

    # 融合方式：
    # - 'concat'：原始做法，直接拼接 short/medium/long 的最后时刻特征
    # - 'gated_short_mlp'：用 short 的特征经小 MLP 产生门控权重，决定吸收多少 medium/long 到 short
    # - 'softmax_short_mlp'：用 short 的特征经小 MLP 对 [short, medium, long] 做 softmax 加权融合
    FUSION_METHOD = 'gated_short_mlp'

    # 融合前分支归一化
    BRANCH_LAYER_NORM = False
    # 'all'：short/medium/long 全归一化
    # 'aux_only'：仅 medium/long 归一化
    # 'none'：不做归一化
    BRANCH_LAYER_NORM_MODE = 'all'

    # gated_short_mlp 参数
    GATE_HIDDEN_SIZE = 16
    GATE_INIT_BIAS = -2.0  # 越小越倾向于初始不吸收（sigmoid 后接近 0）
    GATE_DETACH_AUX = False
    
    HIDDEN_SIZE_LONG = 32    # BiLSTM hidden size
    HIDDEN_SIZE_MEDIUM = 32 # LSTM hidden size
    TCN_CHANNELS = [16, 32, 32] # TCN layers

    D_MODEL = 32 # 最终投射到这个维度，便于融合
    
    # 训练相关    
    # 随机种子
    SEED = 56
    PATIENCE = 20
    DROPOUT = 0.0
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0
    BATCH_SIZE = 32
    EPOCHS = 1000
