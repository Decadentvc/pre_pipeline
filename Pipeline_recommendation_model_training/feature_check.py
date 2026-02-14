#!/usr/bin/env python3
"""
特征有效性验证：
使用树模型判断 dataset_ + model_ 特征
是否能预测 best_candidate_id
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ================= 配置 =================
NUM_CLASSES = 8
SEED = 42
NROWS = 5000   # 和你之前保持一致

# ================= 加载数据 =================
def load_data(path):
    df = pd.read_csv(path, nrows=NROWS)

    dcols = [c for c in df.columns if c.startswith("dataset_")]
    mcols = [c for c in df.columns if c.startswith("model_")]

    df[dcols] = df[dcols].fillna(0)
    df[mcols] = df[mcols].fillna(0)

    Xd = df[dcols].values
    Xm = df[mcols].values

    X = np.concatenate([Xd, Xm], axis=1)

    y = []
    for _, row in df.iterrows():
        try:
            y.append(int(row["best_candidate_id"]) - 1)
        except:
            y.append(0)

    return X, np.array(y), dcols, mcols

# ================= 评估函数 =================
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    top2 = top_k_accuracy_score(y_test, probs, k=2, labels=list(range(NUM_CLASSES)))
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=list(range(NUM_CLASSES)))

    print("\n=== 特征有效性验证结果 ===")
    print(f"Top-1 Acc: {acc:.4f}")
    print(f"Top-2 Acc: {top2:.4f}")
    print(f"Top-3 Acc: {top3:.4f}")

    cm = confusion_matrix(y_test, preds)
    print("\n混淆矩阵（行是真实类别，列是预测类别）:")
    print(cm)

    return acc, top2, top3

# ================= 主流程 =================
def main():
    print("开始特征有效性验证...\n")

    X, y, dcols, mcols = load_data("../HistoryRepo/history_repo_with_features.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")
    print(f"类别分布: {np.bincount(y_train)}\n")

    # 使用 RandomForest 作为强基线
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
