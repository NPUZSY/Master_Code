import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# 将多分类 RNN 模型修改为连续回归模型，输出范围为 [0, 1]
class ActionValueNet(nn.Module):
    # 核心修改点 1: output_dim 默认为 1 (对应连续的回归值)
    def __init__(self, input_dim=7, hidden_dim_rnn=128, hidden_dim_fc=64, output_dim=1):
        super(ActionValueNet, self).__init__()
        
        # --- 隐藏层 1: GRU (7 -> 128) ---
        self.rnn = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim_rnn, 
            num_layers=2, 
            batch_first=True
        )
        
        # --- 隐藏层 2: 全连接层 (128 -> 64) ---
        self.fc_128_64 = nn.Linear(hidden_dim_rnn, hidden_dim_fc)
        # --- 隐藏层 3: 全连接层 (64 -> 64) ---
        self.fc_64_64 = nn.Linear(hidden_dim_fc, hidden_dim_fc)
        
        # --- 输出层: 回归输出 (64 -> 1) ---
        # 核心修改点 2: 输出维度为 output_dim (1)
        self.fc_64_out = nn.Linear(hidden_dim_fc, output_dim)
        self.fc_64_1 = self.fc_64_out 
        
        self.requires_grad_fc_64_1_only = True 

    def forward(self, x):
        x = x.unsqueeze(1) # (N, 1, 7)
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn.squeeze(1) # (N, 128)
        
        # 4.1 隐藏层 2 (64维特征)
        feature_64 = F.relu(self.fc_128_64(out_rnn)) # (N, 64)
        # 4.2 隐藏层 3 (64维特征)
        feature_64 = F.relu(self.fc_64_64(feature_64)) # (64, 64)
        
        # 5. 输出层 (1维原始输出)
        a_raw_out = self.fc_64_out(feature_64) # (N, 1)
        
        # 📌 核心修改 3: 使用 Sigmoid 激活函数将输出约束到 [0, 1] 之间
        a_out = torch.sigmoid(a_raw_out) # (N, 1)
        
        # 返回约束在 [0, 1] 的回归值，以及特征
        return a_out, feature_64

    # 辅助方法: 设置可训练层 (保持不变，但针对新的 1 维输出层)
    def set_trainable_layers(self, trainable=True):
        if self.requires_grad_fc_64_1_only:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.fc_64_out.parameters(): # fc_64_out 是最终输出层
                param.requires_grad = trainable
        else:
            for param in self.parameters():
                param.requires_grad = trainable