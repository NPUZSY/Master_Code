import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 路径设置
def setup_project_root():
    """设置项目根目录到系统路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 网络定义
class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.input = nn.Linear(N_STATES, 64)
        self.input.weight.data.normal_(0, 0.1)
        
        self.lay1 = nn.Linear(64, 64)
        self.lay1.weight.data.normal_(0, 0.1)
        
        self.output = nn.Linear(64, N_ACTIONS)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.lay1(x)
        x = F.relu(x)
        actions_value = self.output(x)
        return actions_value

# 智能体定义
class IndependentDQN(object):
    def __init__(self, agent_name, N_STATES, N_AGENT_ACTIONS, shared_memory=None, memory_counter_ref=None):
        self.agent_name = agent_name
        self.N_AGENT_ACTIONS = N_AGENT_ACTIONS
        self.shared_memory = shared_memory
        self.memory_counter_ref = memory_counter_ref
        
        self.eval_net = Net(N_STATES, N_AGENT_ACTIONS).to(device)
        self.target_net = Net(N_STATES, N_AGENT_ACTIONS).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter = 0
        self.optimizer = None
        self.scheduler = None
        self.loss_func = nn.MSELoss()

    def setup_optimizer(self, lr, lr_factor=0.5, lr_patience=50):
        """设置优化器和学习率调度器（仅训练时使用）"""
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=1e-6
        )

    def load_net(self, path):
        """加载模型权重"""
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.eval_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.eval_net.eval()

    def choose_action(self, state_input: np.ndarray, train=True, epsilon=0.9):
        """选择动作"""
        state_tensor = torch.FloatTensor(state_input).to(device)
        state_tensor = torch.unsqueeze(state_tensor, 0)
        
        if train and np.random.uniform() < epsilon:
            with torch.no_grad():
                actions_value = self.eval_net(state_tensor)
                return torch.max(actions_value, 1)[1].item()
        else:
            with torch.no_grad():
                actions_value = self.eval_net(state_tensor)
                return torch.max(actions_value, 1)[1].item()

    def learn(self, agent_idx, n_states, gamma=0.95, target_replace_iter=100, batch_size=32):
        """训练函数（仅训练时使用）"""
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 采样批次数据
        sample_index = np.random.choice(self.shared_memory.shape[0], batch_size)
        b_memory = self.shared_memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :n_states]).to(device)
        action_col = n_states + agent_idx
        b_a = torch.LongTensor(b_memory[:, action_col:action_col+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, n_states+3:n_states+4]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, n_states+4:]).to(device)

        # 计算Q值
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)

        # 反向传播
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 工具函数
def get_max_folder_name(directory):
    """获取目录中最大的数字文件夹名称"""
    if not os.path.exists(directory):
        return 0
    folders = [int(name) for name in os.listdir(directory) if 
              os.path.isdir(os.path.join(directory, name)) and name.isdigit()]
    return max(folders) + 1 if folders else 0


def font_get():
    # 1. 自动找字体文件（Linux常见路径）
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

    # 2. 加载字体+刷新缓存
    if os.path.exists(font_path):
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        fm.fontManager.addfont(font_path)
        # 3. 全局生效
        plt.rcParams.update({
            'font.sans-serif': ['Times New Roman'],
            'axes.unicode_minus': False  # 解决负号显示
        })