import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse

# ----------------------------------------------------
# 1. 环境与路径配置
# ----------------------------------------------------
def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()
# 从 Chapter3 的引擎导入基础组件
from Scripts.Chapter3.MARL_Engine import Net, IndependentDQN, device

# ----------------------------------------------------
# 2. 适配多任务 RNN 模型结构
# ----------------------------------------------------
class MultiTaskRNN(nn.Module):
    """
    多任务 RNN 结构：处理 7 维输入，输出 1 维回归、4 维分类及 64 维特征
    """
    def __init__(self, input_dim=7, hidden_dim_rnn=256, num_layers_rnn=2, hidden_dim_fc=64):
        super(MultiTaskRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim_rnn, num_layers=num_layers_rnn, batch_first=True)
        self.fc_rnn_to_64 = nn.Linear(hidden_dim_rnn, hidden_dim_fc)
        self.reg_head = nn.Linear(hidden_dim_fc, 1)    # 1维回归输出
        self.cls_head = nn.Linear(hidden_dim_fc, 4)    # 4维分类输出
    
    def forward(self, x):
        # x shape: (batch, 7)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, 7)
        
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn[:, -1, :] # 取最后一个时间步
        
        feature_64 = F.relu(self.fc_rnn_to_64(out_rnn))
        reg_out = self.reg_head(feature_64)
        cls_out = self.cls_head(feature_64)
        
        return reg_out, cls_out, feature_64

# ----------------------------------------------------
# 3. JointNet 类 (模型拼接：RNN + MARL Head)
# ----------------------------------------------------
class JointNet(nn.Module):
    def __init__(self, rnn_part, marl_head):
        super(JointNet, self).__init__()
        self.rnn_part = rnn_part     # 预训练好的 RNN
        self.marl_part = marl_head   # MARL 决策头 (输入维度为 65)

    def forward(self, x):
        # 1. 提取 RNN 特征
        reg_out, _, feature_64 = self.rnn_part(x)
        # 2. 拼接：64维特征 + 1维回归值 = 65维
        joint_input = torch.cat([feature_64, reg_out], dim=1)
        # 3. 传入决策层
        return self.marl_part(joint_input)

    def save_joint_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"✅ JointNet saved to: {path}")

# ----------------------------------------------------
# 4. JointDQN 智能体类
# ----------------------------------------------------
class JointDQN(IndependentDQN):
    def __init__(self, agent_name, rnn_model, n_actions):
        # 初始化基类，输入维度设为 65
        super(JointDQN, self).__init__(agent_name, 65, n_actions)
        
        # 显式保存 n_actions 属性，防止 choose_action 报错
        self.n_actions = n_actions 
        
        # 替换 eval_net 和 target_net 为拼接后的 JointNet
        # 这里的 self.eval_net 是父类生成的 Net(65, n_actions)
        self.eval_net = JointNet(rnn_model, self.eval_net).to(device)
        self.target_net = JointNet(rnn_model, self.target_net).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, x, train=False, epsilon=0.9):
        """
        支持 7 维输入，内部执行 RNN 提取和决策
        注意：参照参考代码，epsilon 是贪婪概率
        """
        x_tensor = torch.FloatTensor(x).to(device)
        if x_tensor.dim() == 1: 
            x_tensor = x_tensor.unsqueeze(0)
            
        # 训练模式下的 Epsilon-Greedy
        # 参考代码逻辑: uniform < epsilon 时利用(贪婪)，否则探索
        if train and np.random.uniform() >= epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                actions_value = self.eval_net(x_tensor)
            action = torch.max(actions_value, 1)[1].item()
            
        return action

# ----------------------------------------------------
# 5. 模型构建与权重迁移
# ----------------------------------------------------
def build_and_test(args):
    # --- A. 加载多任务 RNN ---
    print(f"🚀 Loading RNN Weights from: {args.rnn_path}")
    rnn_model = MultiTaskRNN().to(device)
    rnn_model.load_state_dict(torch.load(args.rnn_path, map_location=device))
    rnn_model.eval()

    # --- B. 准备保存路径 ---
    save_dir = os.path.join(project_root, "nets", "Chap4", "Joint_Net", args.net_date, args.train_id)
    os.makedirs(save_dir, exist_ok=True)

    # --- C. 定义智能体 ---
    agents_info = [
        {"name": "FC", "n_act": 32},
        {"name": "BAT", "n_act": 40},
        {"name": "SC", "n_act": 2}
    ]

    agents = []
    for info in agents_info:
        name, n_act = info["name"], info["n_act"]
        print(f"\nProcessing [{name}] Agent...")

        # 1. 加载旧的 MARL 权重 (原本是 7 维输入)
        marl_file = os.path.join(args.marl_path, f"MARL_Model_{name}.pth")
        if not os.path.exists(marl_file):
            print(f"⚠️  Missing: {marl_file}, skipping.")
            continue

        old_net = Net(N_STATES=7, N_ACTIONS=n_act).to(device)
        old_net.load_state_dict(torch.load(marl_file, map_location=device))

        # 2. 构造新的 Joint 智能体
        agent = JointDQN(name, rnn_model, n_act)

        # 3. 权重迁移 (核心)
        # 迁移 lay1 和 output，input 层(65->64)保持随机初始化
        agent.eval_net.marl_part.lay1.load_state_dict(old_net.lay1.state_dict())
        agent.eval_net.marl_part.output.load_state_dict(old_net.output.state_dict())
        agent.target_net.load_state_dict(agent.eval_net.state_dict())

        # 4. 保存
        agent.eval_net.save_joint_model(os.path.join(save_dir, f"Joint_Model_{name}.pth"))
        agents.append(agent)

    # --- D. 测试 ---
    print("\n" + "="*30)
    print("🔍 Testing Inference with 7-dim input...")
    sample_input = np.random.rand(7).astype(np.float32)
    for a in agents:
        action = a.choose_action(sample_input, train=False)
        print(f"-> Agent [{a.agent_name}] Action: {action}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build JointNet from Pretrained Models")
    parser.add_argument('--rnn_path', type=str, default="/home/siyu/Master_Code/nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth")
    parser.add_argument('--marl_path', type=str, default="./nets/Chap3/1218/36")
    parser.add_argument('--net_date', type=str, default="1219")
    parser.add_argument('--train_id', type=str, default="1")
    args = parser.parse_args()

    build_and_test(args)