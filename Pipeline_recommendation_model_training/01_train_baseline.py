#!/usr/bin/env python3
"""
管道推荐系统 - 阶段1修复版（基于accuracy_ranking评估）
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

# 系统配置
NUM_PIPELINES = 8
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 500

# ================= ranking解析 =================

def parse_ranking_sets(ranking_str):
    """
    将accuracy_ranking解析为 top1/top2/top3 集合
    例如: "1=2>3=4>5>6=7=8"
    """
    if pd.isna(ranking_str):
        return set(), set(), set()

    groups = [g.strip() for g in str(ranking_str).split('>')]
    parsed = []
    for g in groups:
        ids = [int(x)-1 for x in g.split('=') if x.strip().isdigit()]
        parsed.append(set(ids))

    top1 = parsed[0] if len(parsed) >= 1 else set()
    top2 = set().union(*parsed[:2]) if len(parsed) >= 2 else top1
    top3 = set().union(*parsed[:3]) if len(parsed) >= 3 else top2
    return top1, top2, top3


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


def compute_ranking_accuracy(logits, ranking_sets):
    """返回 top1/top2/top3 acc
    规则：
    - Top1: 预测第一名 ∈ 第一名并列集合
    - Top2: 预测第一名 ∈ 前两名并列集合
    - Top3: 预测第一名 ∈ 前三名并列集合
    """
    probs = torch.softmax(logits, dim=1)
    top1_pred = torch.argmax(probs, dim=1).cpu().numpy()

    correct1 = correct2 = correct3 = 0
    for i, p in enumerate(top1_pred):
        r1, r2, r3 = ranking_sets[i]
        if p in r1:
            correct1 += 1
        if p in r2:
            correct2 += 1
        if p in r3:
            correct3 += 1

    n = len(ranking_sets)
    return correct1/n, correct2/n, correct3/n


def train_with_enhanced_monitoring(model, train_loader, val_data, val_ranking_sets, device, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    X_val, y_val = val_data
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    best_val_acc = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_top1, val_top2, val_top3 = compute_ranking_accuracy(val_logits, val_ranking_sets)

        history.append({"epoch":epoch+1,"val_top1":val_top1,"val_top2":val_top2,"val_top3":val_top3})

        if (epoch+1)%20==0 or epoch<5:
            print(f"Epoch {epoch+1:3d}/{epochs} | Top1:{val_top1:.4f} Top2:{val_top2:.4f} Top3:{val_top3:.4f}")

        if val_top1>best_val_acc:
            best_val_acc=val_top1
            torch.save({'model_state_dict':model.state_dict()},'enhanced_pretrained_recommender.pth')

    return model, history


def diagnostic_analysis(model, X_test, ranking_sets, device):
    X_t = torch.FloatTensor(X_test).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        t1,t2,t3 = compute_ranking_accuracy(logits, ranking_sets)

    print("\n=== Ranking评估 ===")
    print(f"Top1 Acc: {t1:.4f}")
    print(f"Top2 Acc: {t2:.4f}")
    print(f"Top3 Acc: {t3:.4f}")
    return t1,t2,t3


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df=pd.read_csv("../HistoryRepo/history_repo_with_features.csv",nrows=5000)

    dcols=[c for c in df.columns if c.startswith('dataset_')]
    mcols=[c for c in df.columns if c.startswith('model_')]

    df[dcols]=df[dcols].fillna(0)
    df[mcols]=df[mcols].fillna(0)

    Xd=StandardScaler().fit_transform(df[dcols].values.astype(np.float32))
    Xm=StandardScaler().fit_transform(df[mcols].values.astype(np.float32))
    X=np.concatenate([Xd,Xm],axis=1)

    y_top1=(df['best_candidate_id'].astype(int)-1).values

    ranking_sets=[parse_ranking_sets(r) for r in df['accuracy_ranking']]

    idx=np.arange(len(X))
    train_idx,test_idx=train_test_split(idx,test_size=0.2,stratify=y_top1,random_state=42)
    train_idx,val_idx=train_test_split(train_idx,test_size=0.125,stratify=y_top1[train_idx],random_state=42)

    X_train,X_val,X_test=X[train_idx],X[val_idx],X[test_idx]
    y_train,y_val=y_top1[train_idx],y_top1[val_idx]

    val_ranking=[ranking_sets[i] for i in val_idx]
    test_ranking=[ranking_sets[i] for i in test_idx]

    train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(y_train)),batch_size=BATCH_SIZE,shuffle=True)

    model=EnhancedRecommender(X.shape[1],NUM_PIPELINES).to(device)
    model,_=train_with_enhanced_monitoring(model,train_loader,(X_val,y_val),val_ranking,device,EPOCHS)

    t1,t2,t3=diagnostic_analysis(model,X_test,test_ranking,device)

    with open('enhanced_stage1_results.json','w') as f:
        json.dump({'top1':t1,'top2':t2,'top3':t3},f,indent=2)

if __name__=='__main__':
    main()
