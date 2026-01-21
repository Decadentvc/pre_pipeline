#!/usr/bin/env python3
"""
阶段2：特征调制门控网络训练
作用：动态调整数据集特征 vs 模型特征的重要性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================= 配置 =================
NUM_PIPELINES = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 300
PRETRAINED_PATH = "enhanced_pretrained_recommender.pth"

# ================= 推荐器（阶段1） =================
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

    def forward(self, x):
        h = self.feature_extractor(x)
        return self.classifier(h)


# ================= 门控网络 =================
class FeatureGatingNet(nn.Module):
    """
    输出 g ∈ [0,1]
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # (N,1)


# ================= 门控系统 =================
class GatedRecommenderSystem(nn.Module):
    def __init__(self, recommender, gating_net, d_dim, m_dim):
        super().__init__()
        self.recommender = recommender
        self.gating = gating_net
        self.d_dim = d_dim
        self.m_dim = m_dim

    def forward(self, x):
        g = self.gating(x)               # (N,1)
        Xd = x[:, :self.d_dim]
        Xm = x[:, self.d_dim:]

        X_mod = torch.cat([
            g * Xd,
            (1 - g) * Xm
        ], dim=1)

        # 关键：不要用 no_grad
        return self.recommender(X_mod)



# ================= 训练函数 =================
def train_gating(system, train_loader, val_data, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(system.gating.parameters(), lr=LEARNING_RATE)

    X_val, y_val = val_data
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)

    best_val_acc = 0
    history = []

    print("\n开始训练特征调制门控网络...")

    for epoch in range(EPOCHS):
        system.train()
        correct, total, total_loss = 0, 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            logits = system(Xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # 验证
        system.eval()
        with torch.no_grad():
            val_logits = system(X_val)
            val_loss = criterion(val_logits, y_val)
            _, val_preds = torch.max(val_logits, 1)
            val_acc = (val_preds == y_val).sum().item() / len(y_val)

            _, top2 = torch.topk(val_logits, 2, dim=1)
            val_top2 = top2.eq(y_val.unsqueeze(1)).sum().item() / len(y_val)

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_top2": val_top2,
            "val_loss": val_loss.item()
        })

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Top2: {val_top2:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(system.gating.state_dict(), "enhanced_stage2_feature_gating.pth")

    return history


# ================= 诊断分析 =================
def diagnostic_analysis(system, X_test, y_test, device):
    X = torch.FloatTensor(X_test).to(device)
    y = torch.LongTensor(y_test).to(device)

    system.eval()
    with torch.no_grad():
        logits = system(X)
        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)

    acc = (preds == y).sum().item() / len(y)

    class_accs = []
    for i in range(NUM_PIPELINES):
        mask = (y == i)
        if mask.sum() > 0:
            class_accs.append((preds[mask] == y[mask]).sum().item() / mask.sum().item())
        else:
            class_accs.append(0.0)

    conf = probs.max(dim=1)[0]
    incorrect = preds != y

    print("\n=== 阶段2 诊断分析（特征调制门控）===")
    print(f"整体准确率: {acc:.4f}")
    print(f"平均置信度: {conf.mean().item():.4f}")

    print("\n各管道准确率:")
    for i, c in enumerate(class_accs):
        print(f"  管道{i+1}: {c:.4f}")

    if incorrect.sum() > 0:
        print(f"\n错误样本数量: {incorrect.sum().item()}")
        print(f"错误样本平均置信度: {conf[incorrect].mean().item():.4f}")

    return acc, class_accs


# ================= 主流程 =================
def main():
    print("="*60)
    print("阶段2：特征调制门控网络")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1. 加载数据
    df = pd.read_csv("../HistoryRepo/history_repo_with_features.csv", nrows=5000)

    dcols = [c for c in df.columns if c.startswith('dataset_')]
    mcols = [c for c in df.columns if c.startswith('model_')]

    df[dcols] = df[dcols].fillna(0)
    df[mcols] = df[mcols].fillna(0)

    Xd = StandardScaler().fit_transform(df[dcols]).astype(np.float32)
    Xm = StandardScaler().fit_transform(df[mcols]).astype(np.float32)

    X = np.concatenate([Xd, Xm], axis=1)
    y = (df["best_candidate_id"].astype(int) - 1).values

    d_dim, m_dim = Xd.shape[1], Xm.shape[1]

    # 2. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 3. 加载推荐器并冻结
    recommender = EnhancedRecommender(X.shape[1], NUM_PIPELINES).to(device)
    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    recommender.load_state_dict(ckpt["model_state_dict"])

    for p in recommender.parameters():
        p.requires_grad = False

    # 4. 门控网络
    gating = FeatureGatingNet(X.shape[1]).to(device)
    system = GatedRecommenderSystem(recommender, gating, d_dim, m_dim).to(device)

    # 5. 训练门控
    history = train_gating(system, train_loader, (X_val, y_val), device)

    # 6. 测试评估
    acc, class_accs = diagnostic_analysis(system, X_test, y_test, device)

    # 7. 保存结果
    results = {
        "final_accuracy": float(acc),
        "class_accuracies": class_accs,
        "training_history": history[-20:]
    }

    with open("enhanced_stage2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n模型已保存:")
    print("  门控网络: enhanced_stage2_feature_gating.pth")
    print("  结果文件: enhanced_stage2_results.json")

    print("\n=== 阶段2结论 ===")
    print("门控网络已学会动态调制数据集特征与模型特征的重要性。")
    print("可进入阶段3进行联合微调。")


if __name__ == "__main__":
    main()
