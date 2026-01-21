#!/usr/bin/env python3
"""
阶段1（改进版）：
- 类别重加权
- Label Smoothing
- 更强特征提取器
- 稳定训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
NUM_PIPELINES = 8
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 400
LABEL_SMOOTHING = 0.1
SEED = 42

# ================== 模型 ==================
class StrongRecommender(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# ================== 数据加载 ==================
def load_data(path, nrows=5000):
    df = pd.read_csv(path, nrows=nrows)

    dcols = [c for c in df.columns if c.startswith("dataset_")]
    mcols = [c for c in df.columns if c.startswith("model_")]

    df[dcols] = df[dcols].fillna(0)
    df[mcols] = df[mcols].fillna(0)

    Xd = df[dcols].values.astype(np.float32)
    Xm = df[mcols].values.astype(np.float32)

    scaler_d = StandardScaler()
    scaler_m = StandardScaler()

    Xd = scaler_d.fit_transform(Xd)
    Xm = scaler_m.fit_transform(Xm)

    X = np.concatenate([Xd, Xm], axis=1)

    y = []
    for _, row in df.iterrows():
        try:
            y.append(int(row["best_candidate_id"]) - 1)
        except:
            y.append(0)

    return X, np.array(y), dcols, mcols

# ================== 采样器 ==================
def create_balanced_loader(X, y, batch_size):
    from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ================== 训练 ==================
def train(model, train_loader, val_data, class_weights, device):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    X_val, y_val = val_data
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)

    best_acc = 0
    history = []

    print("\n开始训练改进版基础推荐器...\n")

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        scheduler.step()

        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

            _, top2 = torch.topk(val_logits, 2, dim=1)
            val_top2 = top2.eq(y_val.unsqueeze(1)).sum().item() / len(y_val)

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_top2": val_top2,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Top2: {val_top2:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1
            }, "stage1_strong_recommender.pth")

    return history

# ================== 评估 ==================
def evaluate(model, X_test, y_test, device):
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    acc = (preds == y_test).float().mean().item()
    conf = probs.max(dim=1)[0].mean().item()
    cm = confusion_matrix(y_test.cpu(), preds.cpu())

    print("\n=== 阶段1改进版 诊断分析 ===")
    print(f"整体准确率: {acc:.4f}")
    print(f"平均置信度: {conf:.4f}")

    for i in range(NUM_PIPELINES):
        mask = (y_test == i)
        if mask.sum() > 0:
            cls_acc = (preds[mask] == y_test[mask]).float().mean().item()
            print(f"  管道{i+1}: {cls_acc:.4f}")

    return acc, cm

# ================== 主函数 ==================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    X, y, dcols, mcols = load_data("../HistoryRepo/history_repo_with_meta_features.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=SEED
    )

    print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

    train_loader = create_balanced_loader(X_train, y_train, BATCH_SIZE)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    model = StrongRecommender(X.shape[1], NUM_PIPELINES).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, train_loader, (X_val, y_val), class_weights, device)

    acc, cm = evaluate(model, X_test, y_test, device)

    with open("stage1_improved_results.json", "w") as f:
        json.dump({
            "final_acc": acc,
            "history": history[-20:]
        }, f, indent=2)

    print("\n模型已保存为: stage1_strong_recommender.pth")

if __name__ == "__main__":
    main()
