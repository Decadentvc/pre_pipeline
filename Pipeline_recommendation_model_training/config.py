# 全局配置参数
class Config:
    # 数据路径
    DATA_PATH = "../HistoryRepo/history_repo_with_features.csv"
    MODEL_SAVE_DIR = "trained_models"
    
    # 特征配置
    DATASET_FEATURE_PREFIX = "dataset_"
    MODEL_FEATURE_PREFIX = "model_"
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE_BASELINE = 1e-3
    LEARNING_RATE_GATE = 1e-4
    LEARNING_RATE_JOINT = 1e-5
    
    # 模型结构
    BASELINE_HIDDEN_DIMS = [128, 64]
    GATE_HIDDEN_DIMS = [64, 32]