import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
    
    def load_and_preprocess(self):
        """加载并预处理数据"""
        df = pd.read_csv(self.config.DATA_PATH)
        
        # 提取特征列
        dcols = [c for c in df.columns if c.startswith(self.config.DATASET_FEATURE_PREFIX)]
        mcols = [c for c in df.columns if c.startswith(self.config.MODEL_FEATURE_PREFIX)]
        
        # 处理缺失值
        df[dcols] = df[dcols].fillna(0)
        df[mcols] = df[mcols].fillna(0)
        
        # 提取特征和目标
        Xd = df[dcols].values.astype(np.float32)
        Xm = df[mcols].values.astype(np.float32)
        
        # 确保目标列存在
        target_col = 'best_accuracy'
        if target_col in df.columns:
            y = df[target_col].values.astype(np.float32)
        else:
            # 如果没有目标列，使用随机值（仅用于测试）
            y = np.random.rand(len(df)).astype(np.float32)
        
        return Xd, Xm, y, dcols, mcols
    
    def create_data_splits(self, Xd, Xm, y, train_ratio=0.8):
        """创建训练/验证分割"""
        n = len(Xd)
        n_train = int(n * train_ratio)
        
        indices = np.random.permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        return (Xd[train_idx], Xm[train_idx], y[train_idx], 
                Xd[val_idx], Xm[val_idx], y[val_idx])