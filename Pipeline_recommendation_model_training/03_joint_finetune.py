#!/usr/bin/env python3
"""
阶段3：推荐器 + 特征门控网络联合微调（Top-3 命中准确率版本）
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
LEARNING_RATE = 5e-4
EPOCHS = 500

PRETRAINED_RECOMMENDER = "enhanced_pretrained_recommender.pth"
PRETRAINED_GATING = "enhanced_stage2_feature_gating.pth"

# ================= 模型定义 =================
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
        return self.classifier(self.feature_extractor(x))


class FeatureGatingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class GatedRecommenderSystem(nn.Module):
    def __init__(self, recommender, gating_net, d_dim):
        super().__init__()
        self.recommender = recommender
        self.gating = gating_net
        self.d_dim = d_dim

    def forward(self, x):
        g = self.gating(x)
        Xd = x[:, :self.d_dim]
        Xm = x[:, self.d_dim:]
        X_mod = torch.cat([g * Xd, (1 - g) * Xm], dim=1)
        return self.recommender(X_mod)


# ================= 训练函数（Top-3 准确率） =================
def train_joint(system, train_loader, val_data, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(system.parameters(), lr=LEARNING_RATE)

    X_val, y_val = val_data
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)

    best_val_acc = 0
    history = []

    print("\n开始联合微调（Top-3 命中准确率）...")

    for epoch in range(EPOCHS):
        system.train()
        total, correct, total_loss = 0, 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            logits = system(Xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            _, top3 = torch.topk(logits, 3, dim=1)
            correct += top3.eq(yb.unsqueeze(1)).any(dim=1).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # ===== 验证 =====
        system.eval()
        with torch.no_grad():
            val_logits = system(X_val)
            val_loss = criterion(val_logits, y_val)

            _, val_top3 = torch.topk(val_logits, 3, dim=1)
            val_acc = val_top3.eq(y_val.unsqueeze(1)).any(dim=1).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss.item()
        })

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "recommender": system.recommender.state_dict(),
                "gating": system.gating.state_dict(),
            }, "enhanced_stage3_joint.pth")

    return history


# ================= 诊断分析（Top-3 命中） =================
def diagnostic_analysis(system, X_test, y_test, device):
    X = torch.FloatTensor(X_test).to(device)
    y = torch.LongTensor(y_test).to(device)

    system.eval()
    with torch.no_grad():
        logits = system(X)
        probs = torch.softmax(logits, dim=1)
        _, top3 = torch.topk(logits, 3, dim=1)

    acc = top3.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()

    class_accs = []
    for i in range(NUM_PIPELINES):
        mask = (y == i)
        if mask.sum() > 0:
            class_accs.append(
                top3[mask].eq(y[mask].unsqueeze(1)).any(dim=1).float().mean().item()
            )
        else:
            class_accs.append(0.0)

    conf = probs.max(dim=1)[0]
    incorrect = ~top3.eq(y.unsqueeze(1)).any(dim=1)

    print("\n=== 阶段3 诊断分析（Top-3 命中）===")
    print(f"整体准确率: {acc:.4f}")
    print(f"平均置信度: {conf.mean().item():.4f}")

    print("\n各管道 Top-3 命中率:")
    for i, c in enumerate(class_accs):
        print(f"  管道{i+1}: {c:.4f}")

    if incorrect.sum() > 0:
        print(f"\n错误样本数量: {incorrect.sum().item()}")
        print(f"错误样本平均置信度: {conf[incorrect].mean().item():.4f}")

    return acc, class_accs


# ================= 主流程 =================
def main():
    print("="*60)
    print("阶段3：联合微调推荐器 + 特征门控网络（Top-3 评估）")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    df = pd.read_csv("../HistoryRepo/history_repo_with_features.csv", nrows=5000)

    dcols = [c for c in df.columns if c.startswith('dataset_')]
    mcols = [c for c in df.columns if c.startswith('model_')]

    df[dcols] = df[dcols].fillna(0)
    df[mcols] = df[mcols].fillna(0)

    Xd = StandardScaler().fit_transform(df[dcols]).astype(np.float32)
    Xm = StandardScaler().fit_transform(df[mcols]).astype(np.float32)

    X = np.concatenate([Xd, Xm], axis=1)
    y = (df["best_candidate_id"].astype(int) - 1).values

    d_dim = Xd.shape[1]

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

    recommender = EnhancedRecommender(X.shape[1], NUM_PIPELINES).to(device)
    ckpt1 = torch.load(PRETRAINED_RECOMMENDER, map_location=device)
    recommender.load_state_dict(ckpt1["model_state_dict"])

    gating = FeatureGatingNet(X.shape[1]).to(device)
    gating.load_state_dict(torch.load(PRETRAINED_GATING, map_location=device))

    system = GatedRecommenderSystem(recommender, gating, d_dim).to(device)

    history = train_joint(system, train_loader, (X_val, y_val), device)

    acc, class_accs = diagnostic_analysis(system, X_test, y_test, device)

    results = {
        "final_top3_accuracy": float(acc),
        "class_top3_accuracies": class_accs,
        "training_history": history[-20:]
    }

    with open("enhanced_stage3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n模型已保存:")
    print("  联合模型: enhanced_stage3_joint.pth")
    print("  结果文件: enhanced_stage3_results.json")

    print("\n=== 阶段3结论 ===")



if __name__ == "__main__":
    main()
