
class ModelConfig:
    DATA_CSV = 'data/GOOGL_processed.csv'

    # FEATURE_COLS = ['Open', 'High', 'Low', 'Close','Open_logret', 'High_logret', 'Low_logret', 'Close_logret', 'Volume']
    # FEATURE_COLS = ['Close', 'Volume']
    FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # TARGET_COL = 'Close_logret'
    TARGET_COL = 'Close'

    INPUT_SIZE = len(FEATURE_COLS)
    OUTPUT_SIZE = 1 

    # 是否归一化
    FEATURE_NORMALIZE_MASK = [True, True, True, True, True]
    # FEATURE_NORMALIZE_MASK = [True, True]
    TARGET_NORMALIZE = True
    
    # 长时序列配置：窗口聚合模式
    LONG_WINDOW_SIZE = 1 # 一个窗口聚合多少天
    LONG_NUM_WINDOWS = 15 # 输入多少个窗口
    SEQ_LEN_LONG = LONG_WINDOW_SIZE * LONG_NUM_WINDOWS # 总回看长度
    
    SEQ_LEN_MEDIUM = 10  # 2个月
    SEQ_LEN_SHORT = 5    # 2周
    
    # 模型维度
    MODEL_MODE = 'full' # 'full', 'long_only', 'medium_only', 'short_only'
    
    HIDDEN_SIZE_LONG = 32    # BiLSTM hidden size
    HIDDEN_SIZE_MEDIUM = 32 # LSTM hidden size
    TCN_CHANNELS = [16, 32] # TCN layers
    
    # Cross Attention 维度
    D_MODEL = 32
    NUM_HEADS = 4
    
    # 训练相关
    PATIENCE = 40
    DROPOUT = 0.3
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 32
    EPOCHS = 100
