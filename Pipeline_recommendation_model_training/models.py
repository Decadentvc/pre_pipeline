import torch.nn as nn
import torch

class BaselineRecommender(nn.Module):
    """基线推荐模型 - 简单的特征拼接"""
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, d_features, m_features):
        # 简单拼接
        concatenated = torch.cat([d_features, m_features], dim=1)
        return self.network(concatenated).squeeze()

class GateNetwork(nn.Module):
    """门控网络"""
    def __init__(self, d_dim, m_dim, hidden_dims):
        super().__init__()
        input_dim = d_dim + m_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, d_features, m_features):
        context = torch.cat([d_features, m_features], dim=1)
        return self.network(context).squeeze()

class FullSystem(nn.Module):
    """完整系统：门控网络 + 推荐模型"""
    def __init__(self, gate_network, recommender):
        super().__init__()
        self.gate = gate_network
        self.recommender = recommender
    
    def forward(self, d_features, m_features):
        # 计算门控值
        g = self.gate(d_features, m_features)
        
        # 调制数据特征
        d_modulated = g.unsqueeze(1) * d_features
        
        # 融合特征并预测
        fused = torch.cat([d_modulated, m_features], dim=1)
        scores = self.recommender(fused)
        
        return scores, g