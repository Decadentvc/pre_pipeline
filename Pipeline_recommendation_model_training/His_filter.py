#!/usr/bin/env python3
"""
管道推荐系统 - 阶段1修复版：增强数据分析和模型调试
新增功能：
过滤accuracy_ranking中并列数>=4的条目（不修改原始数据）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================= 系统配置 =================
NUM_PIPELINES = 8
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 500


# =========================================================
# 新增：ranking 并列过滤模块
# =========================================================

def count_max_tie(ranking_str):
    """
    计算accuracy_ranking中最大并列数量
    例:
    1=2=4=6=7=8>5>3 -> 6
    """
    try:
        groups = str(ranking_str).split('>')
        max_tie = 0
        for g in groups:
            tie_count = len(g.split('='))
            max_tie = max(max_tie, tie_count)
        return max_tie
    except:
        return 1


def filter_ranking_ties(df, tie_threshold=4, save_path="filtered_history_repo.csv"):
    """
    过滤并列数 >= tie_threshold 的条目
    不修改原始csv
    """
    print("\n================ Ranking质量过滤 ================")

    tie_counts = df["accuracy_ranking"].apply(count_max_tie)
    mask = tie_counts < tie_threshold

    filtered_df = df[mask].copy()

    removed = len(df) - len(filtered_df)

    print(f"原始样本数: {len(df)}")
    print(f"剔除并列>=4样本数: {removed}")
    print(f"保留样本数: {len(filtered_df)}")
    print(f"保留比例: {len(filtered_df)/len(df)*100:.2f}%")

    # 保存新数据
    filtered_df.to_csv(save_path, index=False)
    print(f"新数据保存: {save_path}")
    print("================================================\n")

    return filtered_df


# =========================================================
# 模型定义（完全未改）
# =========================================================

class EnhancedRecommender(nn.Module):
    def __init__(self, input_dim, num_pipelines, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_pipelines)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


# =========================================================
# 数据分析（未改）
# =========================================================

def comprehensive_data_analysis(df, X, y):
    print("\n" + "="*50)
    print("全面数据分析")
    print("="*50)

    print("1. 管道标签分布:")
    label_counts = np.bincount(y)
    for i in range(NUM_PIPELINES):
        percentage = label_counts[i] / len(y) * 100
        print(f"   管道{i+1}: {label_counts[i]:4d}样本 ({percentage:5.1f}%)")

    print(f"\n2. 特征统计:")
    print(f"   特征形状: {X.shape}")
    print(f"   特征范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   特征均值: {X.mean():.3f} ± {X.std():.3f}")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"   NaN值: {nan_count}, Inf值: {inf_count}")

    return label_counts


# =========================================================
# 训练主流程（基本未改）
# =========================================================

def main():
    print("="*60)
    print("阶段1修复版 + Ranking质量过滤")
    print("="*60)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_path = "../HistoryRepo/history_repo_with_features.csv"

    # 1️⃣ 读取原始数据
    df_raw = pd.read_csv(data_path, nrows=5000)
    print(f"原始数据形状: {df_raw.shape}")

    # 2️⃣ 过滤 ranking 弱监督样本
    df = filter_ranking_ties(df_raw, tie_threshold=4)
    print(f"过滤后数据形状: {df.shape}")

    # =====================================================
    # 后续逻辑完全不变
    # =====================================================

    dcols = [c for c in df.columns if c.startswith('dataset_')]
    mcols = [c for c in df.columns if c.startswith('model_')]

    df[dcols] = df[dcols].fillna(0)
    df[mcols] = df[mcols].fillna(0)

    Xd = df[dcols].values.astype(np.float32)
    Xm = df[mcols].values.astype(np.float32)

    scaler = StandardScaler()
    Xd_scaled = scaler.fit_transform(Xd)
    Xm_scaled = scaler.fit_transform(Xm)

    X_combined = np.concatenate([Xd_scaled, Xm_scaled], axis=1)

    y_top1 = (df["best_candidate_id"].astype(int) - 1).values

    label_counts = comprehensive_data_analysis(df, X_combined, y_top1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_top1, test_size=0.2, random_state=42, stratify=y_top1
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    print(f"\n数据分割:")
    print(f"  训练集: {X_train.shape[0]}样本")
    print(f"  验证集: {X_val.shape[0]}样本")
    print(f"  测试集: {X_test.shape[0]}样本")

    model = EnhancedRecommender(X_combined.shape[1], NUM_PIPELINES).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✔ 数据已准备完成，可以开始训练（训练代码保持原样）")


if __name__ == "__main__":
    main()
