#!/usr/bin/env python3
"""
管道推荐系统 - 阶段1修复版：增强数据分析和模型调试
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
BATCH_SIZE = 32  # 减小批次大小
LEARNING_RATE = 5e-4  # 调整学习率
EPOCHS = 500

class EnhancedRecommender(nn.Module):
    """增强的推荐模型"""
    def __init__(self, input_dim, num_pipelines, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),  # 使用LeakyReLU避免梯度消失
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_pipelines)
        
        # 改进的权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

def comprehensive_data_analysis(df, X, y):
    """全面的数据分析"""
    print("\n" + "="*50)
    print("全面数据分析")
    print("="*50)
    
    # 1. 标签分布分析
    print("1. 管道标签分布:")
    label_counts = np.bincount(y)
    for i in range(NUM_PIPELINES):
        percentage = label_counts[i] / len(y) * 100
        print(f"   管道{i+1}: {label_counts[i]:4d}样本 ({percentage:5.1f}%)")
    
    # 2. 特征统计分析
    print(f"\n2. 特征统计:")
    print(f"   特征形状: {X.shape}")
    print(f"   特征范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   特征均值: {X.mean():.3f} ± {X.std():.3f}")
    
    # 检查NaN和Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"   NaN值: {nan_count}, Inf值: {inf_count}")
    
    # 3. 特征-目标相关性分析（抽样计算，避免内存问题）
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    try:
        # 计算互信息
        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
        top_features = np.argsort(mi_scores)[-10:]  # 取最重要的10个特征
        print(f"\n3. 特征重要性 (前10个):")
        for i, idx in enumerate(top_features[::-1]):
            print(f"   特征{idx:3d}: {mi_scores[idx]:.4f}")
    except:
        print("\n3. 特征重要性计算跳过（数据可能有问题）")
    
    return label_counts

def create_balanced_dataloader(X, y, batch_size=32):
    """创建平衡的数据加载器（处理类别不平衡）"""
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), 
        torch.LongTensor(y)
    )
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def train_with_enhanced_monitoring(model, train_loader, val_data, device, epochs=200):
    """增强的训练监控"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    X_val, y_val = val_data
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    best_val_acc = 0
    train_history = []
    
    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
        
        # 学习率调度
        scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            _, val_predicted = torch.max(val_logits, 1)
            val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)
            
            # Top-2准确率
            _, top2_pred = torch.topk(val_logits, 2, dim=1)
            val_acc_top2 = top2_pred.eq(y_val_t.unsqueeze(1)).sum().item() / len(y_val_t)
        
        train_acc = correct_predictions / total_samples
        avg_loss = epoch_loss / len(train_loader)
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_loss': val_loss.item(),
            'val_acc': val_acc,
            'val_acc_top2': val_acc_top2,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}, "
                  f"Top2: {val_acc_top2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_acc_top2': val_acc_top2,
                'epoch': epoch + 1
            }, 'enhanced_pretrained_recommender.pth')
    
    return model, train_history

def diagnostic_analysis(model, X_test, y_test, device):
    """诊断分析"""
    print("\n" + "="*50)
    print("模型诊断分析")
    print("="*50)
    
    X_t = torch.FloatTensor(X_test).to(device)
    y_t = torch.LongTensor(y_test).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probabilities = torch.softmax(logits, dim=1)
        _, predictions = torch.max(logits, 1)
        
        # 基本准确率
        accuracy = (predictions == y_t).sum().item() / len(y_t)
        
        # 每个类别的准确率
        class_accuracies = []
        for i in range(NUM_PIPELINES):
            class_mask = (y_t == i)
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == y_t[class_mask]).sum().item() / class_mask.sum().item()
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        # 置信度分析
        max_probs, _ = torch.max(probabilities, 1)
        avg_confidence = max_probs.mean().item()
        
        # 混淆矩阵分析（简化）
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_t.cpu(), predictions.cpu())
        
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均置信度: {avg_confidence:.4f}")
    
    print(f"\n每个管道的准确率:")
    for i in range(NUM_PIPELINES):
        print(f"  管道{i+1}: {class_accuracies[i]:.4f}")
    
    # 分析主要错误模式
    incorrect_mask = (predictions != y_t)
    if incorrect_mask.sum() > 0:
        incorrect_probs = max_probs[incorrect_mask]
        print(f"\n错误分析:")
        print(f"  错误样本平均置信度: {incorrect_probs.mean().item():.4f}")
        print(f"  错误样本数量: {incorrect_mask.sum().item()}")
    
    return accuracy, class_accuracies, cm

def main():
    """主函数"""
    print("="*60)
    print("阶段1修复版: 增强数据分析和调试")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载数据
        data_path = "../HistoryRepo/history_repo_with_features.csv"
        df = pd.read_csv(data_path, nrows=5000)
        print(f"数据形状: {df.shape}")
        
        # 提取特征
        dcols = [c for c in df.columns if c.startswith('dataset_')]
        mcols = [c for c in df.columns if c.startswith('model_')]
        
        # 处理数据
        df[dcols] = df[dcols].fillna(0)
        df[mcols] = df[mcols].fillna(0)
        
        Xd = df[dcols].values.astype(np.float32)
        Xm = df[mcols].values.astype(np.float32)
        
        # 标准化
        scaler = StandardScaler()
        Xd_scaled = scaler.fit_transform(Xd)
        Xm_scaled = scaler.fit_transform(Xm)
        
        X_combined = np.concatenate([Xd_scaled, Xm_scaled], axis=1)
        
        # 解析标签（简化版）
        y_top1 = []
        for idx, row in df.iterrows():
            try:
                best_id = int(row['best_candidate_id']) - 1
                y_top1.append(best_id)
            except:
                y_top1.append(0)
        y_top1 = np.array(y_top1)
        
        # 2. 全面数据分析
        label_counts = comprehensive_data_analysis(df, X_combined, y_top1)
        
        # 3. 数据分割
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
        
        # 4. 创建平衡的数据加载器
        train_loader = create_balanced_dataloader(X_train, y_train, BATCH_SIZE)
        
        # 5. 创建模型
        model = EnhancedRecommender(X_combined.shape[1], NUM_PIPELINES).to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 6. 训练模型
        model, history = train_with_enhanced_monitoring(
            model, train_loader, (X_val, y_val), device, EPOCHS
        )
        
        # 7. 诊断分析
        accuracy, class_accuracies, cm = diagnostic_analysis(model, X_test, y_test, device)
        
        # 8. 保存结果
        results = {
            'final_accuracy': float(accuracy),
            'class_accuracies': [float(acc) for acc in class_accuracies],
            'label_distribution': label_counts.tolist(),
            'training_history': history[-20:]
        }
        
        with open('enhanced_stage1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n改进模型已保存到: enhanced_pretrained_recommender.pth")
        print(f"详细结果已保存到: enhanced_stage1_results.json")
        
        # 9. 建议
        print(f"\n=== 改进建议 ===")
        if accuracy < 0.3:
            print("❌ 性能仍然不理想，建议:")
            print("  1. 检查特征工程: 特征可能无法预测管道选择")
            print("  2. 增加数据量: 当前可能样本不足")
            print("  3. 重新设计特征: 可能需要领域特定的特征")
            print("  4. 简化问题: 考虑减少管道类别或改为回归问题")
        else:
            print("✅ 性能可接受，可以继续阶段2")
        
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()