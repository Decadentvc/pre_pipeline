#!/usr/bin/env python3
"""
特征增强脚本：
1. dataset_* × model_* 交叉特征
2. dataset / model 的统计特征（均值、方差、最大值、最小值）
3. 输出增强特征矩阵 X_enhanced
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ================= 配置 =================
DATA_PATH = "../HistoryRepo/history_repo_with_features.csv"
NROWS = 5000  # 保持和之前一致
OUTPUT_FEATURE_PATH = "X_enhanced.npy"
OUTPUT_FEATURE_CSV = "X_enhanced.csv"

# ================= 加载原始特征 =================
df = pd.read_csv(DATA_PATH, nrows=NROWS)

dcols = [c for c in df.columns if c.startswith("dataset_")]
mcols = [c for c in df.columns if c.startswith("model_")]

df[dcols] = df[dcols].fillna(0)
df[mcols] = df[mcols].fillna(0)

Xd = df[dcols].values.astype(np.float32)
Xm = df[mcols].values.astype(np.float32)

print(f"原始 dataset 特征形状: {Xd.shape}")
print(f"原始 model 特征形状: {Xm.shape}")

# ================= 1. 交叉特征 =================
cross_features = np.einsum('ij,ik->ijk', Xd, Xm)  # shape: (n_samples, n_d, n_m)
cross_features = cross_features.reshape(Xd.shape[0], -1)  # 展平为二维
print(f"交叉特征形状: {cross_features.shape}")

# ================= 2. 统计特征 =================
def compute_stats(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    min_ = X.min(axis=1, keepdims=True)
    max_ = X.max(axis=1, keepdims=True)
    return np.hstack([mean, std, min_, max_])

dataset_stats = compute_stats(Xd)
model_stats = compute_stats(Xm)

print(f"dataset 统计特征形状: {dataset_stats.shape}")
print(f"model 统计特征形状: {model_stats.shape}")

# ================= 3. 拼接最终特征 =================
X_enhanced = np.concatenate([Xd, Xm, cross_features, dataset_stats, model_stats], axis=1)
print(f"增强后特征总形状: {X_enhanced.shape}")

# ================= 4. 标准化 =================
scaler = StandardScaler()
X_enhanced_scaled = scaler.fit_transform(X_enhanced)
print("特征标准化完成")

# ================= 5. 保存 =================
np.save(OUTPUT_FEATURE_PATH, X_enhanced_scaled)
pd.DataFrame(X_enhanced_scaled).to_csv(OUTPUT_FEATURE_CSV, index=False)
print(f"增强特征已保存为: {OUTPUT_FEATURE_PATH} 和 {OUTPUT_FEATURE_CSV}")

# ================= 6. 标签处理 =================
y = []
for _, row in df.iterrows():
    try:
        y.append(int(row["best_candidate_id"]) - 1)
    except:
        y.append(0)
y = np.array(y)
np.save("y.npy", y)
print("标签已保存为 y.npy")
