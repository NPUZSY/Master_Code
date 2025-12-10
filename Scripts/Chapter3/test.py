import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.patches as mpatches
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# 导入环境
from Scripts.Env import Envs

# ====================================================================
# ❗ 必须重新定义网络和智能体类，以正确加载 MARL 模型
# ====================================================================
N_STATES = 7  # [load, temp, P_fc, P_bat, P_sc, SOC_b, SOC_sc]
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self, N_ACTIONS):
        super(Net, self).__init__()
        self.input = nn.Linear(N_STATES, 64)
        self.lay1 = nn.Linear(64, 64)
        self.output = nn.Linear(64, N_ACTIONS)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.lay1(x)
        x = F.relu(x)
        actions_value = self.output(x)
        return actions_value


class IndependentDQN(object):
    def __init__(self, agent_name, N_AGENT_ACTIONS):
        self.agent_name = agent_name
        self.N_AGENT_ACTIONS = N_AGENT_ACTIONS
        self.eval_net = Net(N_AGENT_ACTIONS).to(device)

    def load_net(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.eval_net.eval()

    def choose_action(self, state_input: np.ndarray, train=False):
        temp = torch.FloatTensor(state_input)
        state_input = torch.unsqueeze(temp.to(device), 0)
        with torch.no_grad():
            actions_value = self.eval_net.forward(state_input)
            action_index = torch.max(actions_value, 1)[1].item()
        return action_index


# ====================================================================
# 初始化环境和模型加载
# ====================================================================
env = Envs()

net_data = '1210'
train_id = '6'
net_name_base = 'bs64_lr20_ep_1732_pool1000_freq10_MARL_MARL_IQL_32x20x2_MAX_R-6688'

FC_Agent = IndependentDQN("FC_Agent", N_FC_ACTIONS)
Bat_Agent = IndependentDQN("Bat_Agent", N_BAT_ACTIONS)
SC_Agent = IndependentDQN("SC_Agent", N_SC_ACTIONS)

BASE_PATH = f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}"
try:
    FC_Agent.load_net(f"{BASE_PATH}_FC.pth")
    Bat_Agent.load_net(f"{BASE_PATH}_BAT.pth")
    SC_Agent.load_net(f"{BASE_PATH}_SC.pth")
    print(f"Successfully loaded MARL models from: {BASE_PATH}_*.pth")
except FileNotFoundError:
    print(f"Error: One or more model files not found in {BASE_PATH}_*.pth")
    raise

# reset env
s = env.reset()
step = 0

# --------------------------------------------------------------------
# 记录数据列表
# --------------------------------------------------------------------
power_fc = []
battery_power = []
power_sc = []  # 超级电容功率
soc_bat = []
soc_sc_list = []
times = []
loads = env.loads[:-1]
temperature = env.temperature[:-1]

time_start = time.time()
ep_r = 0

# --------------------------------------------------------------------
# 等效氢耗累计（分别为 FC 与 Battery）
# --------------------------------------------------------------------
total_equivalent_H2_consumption = 0.0
total_fc_H2_g = 0.0
total_bat_H2_g = 0.0

# --------------------------------------------------------------------
# 超级电容吸收/释放能量统计（按功率和时间积分）
# 我们统计：release (P_sc>0) 与 absorb (P_sc<0) 的能量（J）与电量（Wh）
# --------------------------------------------------------------------
sc_release_power_sum = 0.0  # ∑ P_sc for P_sc>0 (W·s if dt=1)
sc_absorb_power_sum = 0.0   # ∑ (-P_sc) for P_sc<0 (W·s)
# We'll convert to Wh later using dt

# --------------------------------------------------------------------
# 电池充电（能量回收）状态统计
# --------------------------------------------------------------------
bat_charge_steps = 0  # battery in charging state (P_bat < 0) 的步数
total_steps = 0

# --------------------------------------------------------------------
# 计时细分（累加每一步各个阶段耗时）
# --------------------------------------------------------------------
episode_times = {
    'Action_Select': 0.0,
    'Env_Step': 0.0,
    'Logging_Processing': 0.0,
    'Other_Overhead': 0.0
}

# 其他参数
sc_inactive_threshold = 1e-3  # 认为 SC 未参与的阈值 (W)
dt = getattr(env, "dt", 1.0)  # 环境时间步长（s），若 env 没有，默认 1s

# --------------------------------------------------------------------
# 主循环（带细分计时与统计）
# --------------------------------------------------------------------
while True:
    t0_loop = time.time()

    # 1) Action selection 计时
    t_as0 = time.time()
    a_fc = FC_Agent.choose_action(s, train=False)
    a_bat = Bat_Agent.choose_action(s, train=False)
    a_sc = SC_Agent.choose_action(s, train=False)
    action_list = [a_fc, a_bat, a_sc]
    t_as1 = time.time()
    episode_times['Action_Select'] += (t_as1 - t_as0)

    # 2) Env step 计时
    t_env0 = time.time()
    s_, r, done, info = env.step(action_list)
    t_env1 = time.time()
    episode_times['Env_Step'] += (t_env1 - t_env0)

    # ----------------------------------------------------------------
    # 等效氢耗累加（分别记录 FC 与 BAT 的贡献）
    # info 中预期包含 "C_fc_g" 和 "C_bat_g"（克）
    fc_H2_g = float(info.get("C_fc_g", 0.0))
    bat_H2_g = float(info.get("C_bat_g", 0.0))
    total_fc_H2_g += fc_H2_g
    total_bat_H2_g += bat_H2_g
    H2_step_g = fc_H2_g + bat_H2_g
    total_equivalent_H2_consumption += H2_step_g
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 记录功率 / SOC 列表（使用 s_）
    # s_ 结构: [next_load, next_temp, P_fc, P_bat, P_sc, SOC_b, SOC_sc]
    times.append(step)
    power_fc.append(float(s_[2]))
    battery_power.append(float(s_[3]))
    power_sc.append(float(s_[4]))
    soc_bat.append(float(s_[5]))
    soc_sc_list.append(float(s_[6]))

    # 统计超级电容吸收释放（按功率积分）
    p_sc = float(s_[4])
    if p_sc > 0:
        sc_release_power_sum += p_sc * dt  # W * s -> J
    elif p_sc < 0:
        sc_absorb_power_sum += (-p_sc) * dt  # store positive J

    # 统计电池充电步数（认为 P_bat < 0 为充电/能量回收）
    if float(s_[3]) < 0:
        bat_charge_steps += 1

    ep_r += r

    # 记录处理时间（Logging/Processing）
    t_log0 = time.time()
    # （这里可能会有额外的后处理；我们把计时包括在 Logging_Processing）
    t_log1 = time.time()
    episode_times['Logging_Processing'] += (t_log1 - t_log0)

    t1_loop = time.time()
    episode_times['Other_Overhead'] += (t1_loop - t0_loop) - (
        (t_as1 - t_as0) + (t_env1 - t_env0) + (t_log1 - t_log0)
    )

    total_steps += 1

    if done:
        break

    s = s_
    step += 1
    if step >= len(loads):
        break

time_finish = time.time()

# --------------------------------------------------------------------
# 汇总计算
# --------------------------------------------------------------------
total_steps = max(1, total_steps)
total_time = time_finish - time_start

# 1) 等效氢耗分解与比重
fc_h2 = total_fc_H2_g
bat_h2 = total_bat_H2_g
total_h2 = total_equivalent_H2_consumption if total_equivalent_H2_consumption > 0 else (fc_h2 + bat_h2)
fc_h2_ratio = fc_h2 / total_h2 if total_h2 > 0 else 0.0
bat_h2_ratio = bat_h2 / total_h2 if total_h2 > 0 else 0.0

# 2) 电池能量回收时间与比重
bat_charge_time_s = bat_ch
