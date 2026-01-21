#!/usr/bin/env python3
"""
增强特征 Baseline 验证
使用 XGBoost / LightGBM 风格树模型
评估 Top-1 / Top-2 / Top-3 准确率
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ================= 配置 =================
NUM_CLASSES = 8
SEED = 42

# ================= 加载增强特征 =================
X = np.load("X_enhanced.npy")
y = np.load("y.npy")

print(f"增强特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

# ================= 数据分割 =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")
print(f"类别分布: {np.bincount(y_train)}\n")

# ================= 模型训练 =================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

print("开始训练 RandomForest 模型...")
model.fit(X_train, y_train)

# ================= 评估函数 =================
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=list(range(NUM_CLASSES)))
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=list(range(NUM_CLASSES)))

    print("\n=== 增强特征 Baseline 结果 ===")
    print(f"Top-1 Acc: {acc:.4f}")
    print(f"Top-2 Acc: {top2:.4f}")
    print(f"Top-3 Acc: {top3:.4f}")

    cm = confusion_matrix(y_test, preds)
    print("\n混淆矩阵（行是真实类别，列是预测类别）:")
    print(cm)

    return acc, top2, top3

# ================= 评估 =================
evaluate(model, X_test, y_test)
