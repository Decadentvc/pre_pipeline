# debug_training.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def debug_data_and_model():
    print("=== 训练问题诊断 ===")
    
    # 1. 检查数据统计
    df = pd.read_csv("../HistoryRepo/history_repo_with_features.csv")
    print(f"数据形状: {df.shape}")
    
    # 检查数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数值列数量: {len(numeric_cols)}")
    
    # 检查目标列
    if 'best_accuracy' in df.columns:
        target_stats = df['best_accuracy'].describe()
        print(f"目标列统计:\n{target_stats}")
        
        # 检查异常值
        q1 = df['best_accuracy'].quantile(0.25)
        q3 = df['best_accuracy'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df['best_accuracy'] < lower_bound) | (df['best_accuracy'] > upper_bound)]
        print(f"目标列异常值数量: {len(outliers)}")
    
    # 2. 检查特征尺度
    print("\n=== 特征统计 ===")
    for col in numeric_cols[:10]:  # 只检查前10个特征
        if df[col].dtype in [np.int64, np.float64]:
            stats = df[col].describe()
            print(f"{col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}]")
    
    return df

def create_safe_model():
    """创建安全的模型配置"""
    class SafeBaselineRecommender(nn.Module):
        def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1):
            super().__init__()
            
            # 更小的网络，更好的初始化
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
            
            # 安全的权重初始化
            self._initialize_weights()
        
        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
        
        def forward(self, d_features, m_features):
            # 简单的特征拼接
            concatenated = torch.cat([d_features, m_features], dim=1)
            return self.network(concatenated).squeeze()
    
    return SafeBaselineRecommender

if __name__ == "__main__":
    debug_data_and_model()