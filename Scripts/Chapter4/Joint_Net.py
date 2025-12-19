import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
        sys.path.append(project_root)

from Scripts.utils.global_utils import *
from Scripts.Chapter3.MARL_Engine import (
    device,
    IndependentDQN,
    Net
)

from Scripts.Chapter4.RNN import ActionValueNet  # 修正RNN模块导入路径（根据项目树）


# 耦合网络定义（RNN + MARL）
class JointNet(nn.Module):
    def __init__(self, rnn_input_dim=7, rnn_hidden_dim_rnn=128, num_layers_rnn=2,
                 rnn_hidden_dim_fc=64, rnn_output_dim_reg=1, rnn_output_dim_cls=64,
                 marl_n_actions=32):
        super(JointNet, self).__init__()
        # RNN子网络（复用test_RNN.py中的结构）
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim_rnn,
            num_layers=num_layers_rnn,
            batch_first=True
        )
        self.rnn_fc_rnn_to_64 = nn.Linear(rnn_hidden_dim_rnn, rnn_hidden_dim_fc)
        self.rnn_reg_head = nn.Linear(rnn_hidden_dim_fc, rnn_output_dim_reg)
        self.rnn_cls_head = nn.Linear(rnn_hidden_dim_fc, rnn_output_dim_cls)
        
        # MARL子网络（输入维度改为65=64+1）
        self.marl_input = nn.Linear(65, 64)  # 64维RNN特征 + 1维RNN回归输出
        self.marl_input.weight.data.normal_(0, 0.1)
        
        self.marl_lay1 = nn.Linear(64, 64)
        self.marl_lay1.weight.data.normal_(0, 0.1)
        
        self.marl_output = nn.Linear(64, marl_n_actions)
        self.marl_output.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # RNN部分前向传播
        x_rnn = x.unsqueeze(1)  # (N, 1, 7)
        out_rnn, _ = self.rnn(x_rnn)
        feature_rnn = out_rnn.squeeze(1)  # (N, 128)
        feature_64 = F.relu(self.rnn_fc_rnn_to_64(feature_rnn))  # (N, 64)
        
        # RNN输出（回归+分类）
        a_out_reg = torch.sigmoid(self.rnn_reg_head(feature_64))  # (N, 1)
        a_out_cls_logits = self.rnn_cls_head(feature_64)  # (N, 64)
        
        # 特征拼接（64维特征 + 1维回归输出）
        marl_input = torch.cat([feature_64, a_out_reg], dim=1)  # (N, 65)
        
        # MARL部分前向传播
        x_marl = self.marl_input(marl_input)
        x_marl = F.relu(x_marl)
        x_marl = self.marl_lay1(x_marl)
        x_marl = F.relu(x_marl)
        actions_value = self.marl_output(x_marl)  # 最终动作价值
        
        return actions_value, a_out_reg, a_out_cls_logits, feature_64

    def load_joint_weights(self, rnn_model_path, marl_model_path=None):
        """
        加载联合网络的权重（支持分别加载RNN和MARL的预训练参数）
        
        Args:
            rnn_model_path (str): RNN模型权重文件路径 (.pth)
            marl_model_path (str, optional): MARL模型权重文件路径 (.pth)
        """
        # 1. 加载RNN部分权重（复用ActionValueNet的结构）
        if os.path.exists(rnn_model_path):
            # 初始化RNN参考模型
            rnn_ref = ActionValueNet(
                input_dim=self.rnn.input_size,
                hidden_dim_rnn=self.rnn.hidden_size,
                num_layers_rnn=self.rnn.num_layers,
                hidden_dim_fc=self.rnn_fc_rnn_to_64.out_features,
                output_dim_reg=self.rnn_reg_head.out_features,
                output_dim_cls=self.rnn_cls_head.out_features
            )
            # 加载权重
            rnn_ref.load_state_dict(torch.load(rnn_model_path, map_location=device))
            # 复制到当前网络的RNN部分
            self.rnn.load_state_dict(rnn_ref.rnn.state_dict())
            self.rnn_fc_rnn_to_64.load_state_dict(rnn_ref.fc_rnn_to_64.state_dict())
            self.rnn_reg_head.load_state_dict(rnn_ref.reg_head.state_dict())
            self.rnn_cls_head.load_state_dict(rnn_ref.cls_head.state_dict())
            print(f"✅ 成功加载RNN权重: {rnn_model_path}")
        else:
            raise FileNotFoundError(f"RNN模型文件不存在: {rnn_model_path}")

        # 2. 加载MARL部分权重（可选，若提供）
        if marl_model_path and os.path.exists(marl_model_path):
            # 假设MARL模型是IndependentDQN中使用的Net结构
            marl_ref = Net(
                N_STATES=65,  # 联合网络中MARL的输入维度是65
                N_ACTIONS=self.marl_output.out_features
            )
            marl_ref.load_state_dict(torch.load(marl_model_path, map_location=device))
            # 复制到当前网络的MARL部分
            self.marl_input.load_state_dict(marl_ref.input.state_dict())
            self.marl_lay1.load_state_dict(marl_ref.lay1.state_dict())
            self.marl_output.load_state_dict(marl_ref.output.state_dict())
            print(f"✅ 成功加载MARL权重: {marl_model_path}")


class JointDQN(IndependentDQN):
    """继承自IndependentDQN，扩展支持JointNet作为动作选择网络"""
    
    def __init__(self, agent_name, N_STATES, N_AGENT_ACTIONS, shared_memory=None, 
                 memory_counter_ref=None, joint_net_kwargs=None):
        # 调用父类初始化方法（保持原有参数兼容）
        super().__init__(agent_name, N_STATES, N_AGENT_ACTIONS, shared_memory, memory_counter_ref)
        
        # 配置联合网络参数（默认值基于test_RNN.py中的ActionValueNet）
        self.joint_net_kwargs = joint_net_kwargs or {
            "input_dim": N_STATES,
            "hidden_dim_rnn": 128,
            "num_layers_rnn": 2,
            "hidden_dim_fc": 64,
            "output_dim_reg": 1,
            "output_dim_cls": N_AGENT_ACTIONS  # 分类输出维度与动作数匹配
        }
        
        # 替换为JointNet（复用test_RNN.py中的ActionValueNet作为联合网络）
        self.eval_net = ActionValueNet(**self.joint_net_kwargs).to(device)
        self.target_net = ActionValueNet(** self.joint_net_kwargs).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def load_net(self, base_name, pretrain_date=None, pretrain_train_id=None, 
                 rnn_base_name=None, project_root=None):
        """
        重写加载方法：支持传入base_name，根据智能体名称拼接真实模型路径
        适配多智能体（FC_Agent/Bat_Agent/SC_Agent）的模型命名规则
        
        Args:
            base_name (str): 模型前缀（如MARL_Model）
            pretrain_date (str, optional): 预训练日期目录（如1218）
            pretrain_train_id (str, optional): 预训练训练ID目录（如9）
            rnn_base_name (str, optional): RNN模型前缀（如rnn_classifier_multitask）
            project_root (str, optional): 项目根目录，默认从MARL_Engine获取
        """
        # 1. 确定项目根目录

        # 2. 拼接智能体对应的模型后缀
        agent_suffix_map = {
            "FC_Agent": "FC",
            "Bat_Agent": "BAT",
            "SC_Agent": "SC"
        }
        if self.agent_name not in agent_suffix_map:
            raise ValueError(f"不支持的智能体名称: {self.agent_name}，仅支持{list(agent_suffix_map.keys())}")
        agent_suffix = agent_suffix_map[self.agent_name]
        
        # 3. 拼接MARL部分模型路径
        if pretrain_date and pretrain_train_id:
            marl_base_dir = os.path.join(project_root, "nets", "Chap3", pretrain_date, pretrain_train_id)
            marl_model_path = os.path.join(marl_base_dir, f"{base_name}_{agent_suffix}.pth")
        else:
            # 若未指定日期/ID，直接使用base_name作为完整路径
            marl_model_path = base_name
        
        # 4. 拼接RNN部分模型路径（可选）
        rnn_model_path = None
        if rnn_base_name:
            # RNN模型默认存储路径（可根据实际目录调整）
            rnn_base_dir = os.path.join(project_root, "nets", "Chap4", "RNN_Reg_Opt_MultiTask", pretrain_date, pretrain_train_id)
            rnn_model_path = os.path.join(rnn_base_dir, f"{rnn_base_name}.pth")

        # 5. 检查MARL模型文件是否存在
        if not os.path.exists(marl_model_path):
            raise FileNotFoundError(f"MARL模型文件不存在: {marl_model_path}")
        
        # 6. 加载权重（兼容单独加载RNN权重或完整联合网络权重）
        if rnn_model_path and os.path.exists(rnn_model_path):
            # 先加载RNN基础权重
            rnn_state_dict = torch.load(rnn_model_path, map_location=device)
            # 过滤出RNN相关层权重（仅加载RNN部分）
            rnn_filtered = {k: v for k, v in rnn_state_dict.items() if k.startswith('rnn.')}
            self.eval_net.load_state_dict(rnn_filtered, strict=False)
            print(f"✅ 成功加载{self.agent_name}的RNN权重: {rnn_model_path}")
            
            # 再加载MARL部分权重
            marl_state_dict = torch.load(marl_model_path, map_location=device)
            marl_filtered = {k: v for k, v in marl_state_dict.items() if not k.startswith('rnn.')}
            self.eval_net.load_state_dict(marl_filtered, strict=False)
            print(f"✅ 成功加载{self.agent_name}的MARL权重: {marl_model_path}")
        else:
            # 直接加载完整联合网络权重
            full_state_dict = torch.load(marl_model_path, map_location=device)
            self.eval_net.load_state_dict(full_state_dict)
            print(f"✅ 成功加载{self.agent_name}的联合网络权重: {marl_model_path}")
        
        # 7. 同步目标网络并设置为评估模式
        self.eval_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.eval_net.eval()

    def choose_action(self, state_input: np.ndarray, train=True, epsilon=0.9):
        """重写动作选择：适配JointNet的输出格式（从test_RNN.py兼容）"""
        state_tensor = torch.FloatTensor(state_input).to(device)
        state_tensor = torch.unsqueeze(state_tensor, 0)  # 增加批次维度
        
        with torch.no_grad():
            # JointNet输出格式：(a_out_reg, a_out_cls_logits, feature_64)
            _, actions_logits, _ = self.eval_net(state_tensor)
            actions_value = F.softmax(actions_logits, dim=1)  # 转换为概率分布
            
            # 保持原有epsilon-greedy策略
            if train and np.random.uniform() < epsilon:
                return torch.max(actions_value, 1)[1].item()
            else:
                return torch.max(actions_value, 1)[1].item()

    def learn(self, agent_idx, n_states, gamma=0.95, target_replace_iter=100, batch_size=32):
        """重写训练方法：适配JointNet的输出格式"""
        # 复用父类的目标网络同步逻辑
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 复用父类的经验采样逻辑
        sample_index = np.random.choice(self.shared_memory.shape[0], batch_size)
        b_memory = self.shared_memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :n_states]).to(device)
        action_col = n_states + agent_idx
        b_a = torch.LongTensor(b_memory[:, action_col:action_col+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, n_states+3:n_states+4]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, n_states+4:]).to(device)

        # 适配JointNet的Q值计算（提取分类头输出）
        _, q_eval_logits, _ = self.eval_net(b_s)
        q_eval = F.softmax(q_eval_logits, dim=1).gather(1, b_a)
        
        _, q_next_logits, _ = self.target_net(b_s_)
        q_next = F.softmax(q_next_logits, dim=1).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)

        # 复用父类的反向传播逻辑
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    N_STATES = 7
    N_FC_ACTIONS = 32
    N_BAT_ACTIONS = 40
    N_SC_ACTIONS = 2
    N_FC_ACTIONS

    fc_agent_old = IndependentDQN("FC_Agent", N_STATES, N_FC_ACTIONS)

    # 初始化JointDQN智能体
    fc_agent = JointDQN(
        agent_name="FC_Agent",
        N_STATES=7,
        N_AGENT_ACTIONS=32,
        joint_net_kwargs={"hidden_dim_rnn": 256}
    )

    # 加载预训练模型（传入base_name和目录参数）
    fc_agent.load_net(
        base_name="MARL_Model",
        pretrain_date="1218",
        pretrain_train_id="36",
        rnn_base_name="MARL_Model",
        project_root=project_root
    )
    pass

if __name__ == '__main__':
    main()