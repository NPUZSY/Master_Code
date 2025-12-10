import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.patches as mpatches


from Scripts.Env import Envs

# ====================================================================
# ❗ 必须重新定义网络和智能体类，以正确加载 MARL 模型
# 假设参数与 train_MARL.py 保持一致
# ====================================================================
# 环境状态和动作空间常量 (基于 train_MARL.py 和 Env.py 的结构)
N_STATES = 7  # [load, temp, P_fc, P_bat, P_sc, SOC_b, SOC_sc]
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


class Net(nn.Module):
    """
    通用 Q-网络结构 (与训练代码保持一致)
    """

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
    """
    Independent DQN (I-DQN) 智能体类（简化版，仅用于加载和选择动作）
    """

    def __init__(self, agent_name, N_AGENT_ACTIONS):
        self.agent_name = agent_name
        self.N_AGENT_ACTIONS = N_AGENT_ACTIONS
        self.eval_net = Net(N_AGENT_ACTIONS).to(device)

    def load_net(self, path):
        # 允许在 CPU/GPU 之间加载
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.eval_net.eval()  # 切换到评估模式

    def choose_action(self, state_input: np.ndarray, train=False):
        # 确保输入是正确的 Tensor 格式
        temp = torch.FloatTensor(state_input)
        state_input = torch.unsqueeze(temp.to(device), 0)

        with torch.no_grad():
            actions_value = self.eval_net.forward(state_input)
            # 选择 Q 值最大的局部动作索引
            action_index = torch.max(actions_value, 1)[1].item()
        return action_index


# ====================================================================


# 初始化环境
env = Envs()

# --------------------------------------------------------------------
# ❗ MARL 模型加载
# --------------------------------------------------------------------
net_data = '1210'  # 您的日期目录
train_id = '0'  # 您的训练 ID 目录
# ❗ 注意：这里需要替换为您实际训练保存的模型名称基础
# 假设您保存的模型文件名为: {BASE_NAME}_FC.pth, {BASE_NAME}_BAT.pth, {BASE_NAME}_SC.pth
net_name_base = 'bs64_lr50_ep_188_pool10_freq10_MARL_MARL_IQL_32x20x2_MAX_R-15'

# 实例化三个独立的 DQN 智能体
FC_Agent = IndependentDQN("FC_Agent", N_FC_ACTIONS)
Bat_Agent = IndependentDQN("Bat_Agent", N_BAT_ACTIONS)
SC_Agent = IndependentDQN("SC_Agent", N_SC_ACTIONS)

# 加载模型
BASE_PATH = f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}"
try:
    FC_Agent.load_net(f"{BASE_PATH}_FC.pth")
    Bat_Agent.load_net(f"{BASE_PATH}_BAT.pth")
    SC_Agent.load_net(f"{BASE_PATH}_SC.pth")
    print(f"Successfully loaded MARL models from: {BASE_PATH}_*.pth")
except FileNotFoundError:
    print(f"Error: One or more model files not found in {BASE_PATH}_*.pth")
    # 退出或使用默认网络
    raise

s = env.reset()
step = 0
# --------------------------------------------------------------------
# ❗ 记录数据列表更新，新增超级电容功率和 SOC
# --------------------------------------------------------------------
power_fc = []
battery_power = []
power_sc = []  # 超级电容功率
soc_bat = []  # 电池 SOC (原名 soc)
soc_sc_list = []  # 超级电容 SOC
times = []
loads = env.loads[:-1]  # 负载 profile
temperature = env.temperature[:-1]  # 温度 profile

time_start = time.time()
ep_r = 0  # 初始化 ep_r

# --------------------------------------------------------------------
# ❗ 运行主循环
# --------------------------------------------------------------------
while True:
    # ❗ MARL: 三个智能体独立选择局部动作 (train=False)
    a_fc = FC_Agent.choose_action(s, train=False)
    a_bat = Bat_Agent.choose_action(s, train=False)
    a_sc = SC_Agent.choose_action(s, train=False)
    action_list = [a_fc, a_bat, a_sc]  # 组合动作

    # take action
    s_, r, done, _ = env.step(action_list)

    # --------------------------------------------------------------------
    # ❗ 数据记录：使用新状态 s_ (t 时刻动作产生的结果) 记录所有组件功率和 SOC
    # s_ 结构: [next_load, next_temp, P_fc, P_bat, P_sc, SOC_b, SOC_sc]
    # --------------------------------------------------------------------
    times.append(step)
    power_fc.append(s_[2])
    battery_power.append(s_[3])
    power_sc.append(s_[4])  # 记录超级电容功率
    soc_bat.append(s_[5])  # 记录电池 SOC
    soc_sc_list.append(s_[6])  # 记录超级电容 SOC

    ep_r += r

    if done:
        break

    s = s_
    step += 1
    # 保证 loads/temperature 列表不越界
    if step >= len(loads):
        break

time_finish = time.time()

# --------------------------------------------------------------------
# ❗ 绘图部分更新
# --------------------------------------------------------------------
# 定义颜色列表
best_color = ['#3570a8', '#f09639', '#42985e', '#c84343', '#8a7ab5']
article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
colors = article_color
LINES_ALPHA = 1
LABEL_FONT_SIZE = 18

# 创建主坐标轴
fig, ax1 = plt.subplots(figsize=(15, 5))

fig.subplots_adjust(
    top=0.965,
    bottom=0.125,
    left=0.085,
    right=0.875,
    hspace=0.2,
    wspace=0.2
)

# 功率曲线 (ax1)
l1, = ax1.plot(times, loads, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
l2, = ax1.plot(times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
l3, = ax1.plot(times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
l6, = ax1.plot(times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)  # ❗ SC 功率

# 设置主坐标轴的标签和颜色
ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
ax1.set_ylabel('Power/W', color='black', fontsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='x', labelcolor='black', labelsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='y', labelcolor='black', labelsize=LABEL_FONT_SIZE)

# 创建第二个 y 轴 (SOCs)
ax2 = ax1.twinx()
# ax2.spines['right'].set_position(('outward', 20))
l4, = ax2.plot(times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
l7, = ax2.plot(times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)  # ❗ SC SOC

# 设置第二个 y 轴的标签和颜色
ax2.set_ylabel('SOC', color='black', fontsize=LABEL_FONT_SIZE)  # ❗ SOC 统一标签
ax2.tick_params(axis='y', labelcolor='black', labelsize=LABEL_FONT_SIZE)

# 创建第三个 y 轴 (Temperature)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 65))
l5, = ax3.plot(times, temperature, label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)

# 设置第三个 y 轴的标签和颜色
ax3.set_ylabel('Environment Temperature/℃', color=colors[4], fontsize=LABEL_FONT_SIZE)
ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)

# 合并所有坐标轴的图例
lines = [l1, l2, l3, l6, l4, l7, l5]  # ❗ 更新图例列表
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower center', ncol=3)  # 调整图例位置和列数

# 设置横轴和纵轴显示范围
plt.xlim(0, 600)
# ❗ 移除 plt.ylim(30, -100)，避免覆盖 Power 轴或 SOC 轴的自动缩放
# plt.ylim(30, -100) # 原代码注释，如果需要固定温度轴范围，请取消注释并确认范围

# 起飞阶段背景和数据
ax1.axvspan(0, 150, alpha=0.2, color='lightblue', label='Taking off & Climbing')
# 巡航阶段背景和数据
ax1.axvspan(150, 450, alpha=0.2, color='lightgreen', label='Cruising')
# 降落阶段背景和数据
ax1.axvspan(450, 600, alpha=0.2, color='salmon', label='Descending & underwater')

# 添加网格线
ax1.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

# 重新组织图例
taking_off_patch = mpatches.Patch(color='lightblue', label='Air flight', alpha=0.2)
cruising_patch = mpatches.Patch(color='lightgreen', label='Surface navigation', alpha=0.2)
underwater_patch = mpatches.Patch(color='salmon', label='Underwater navigation', alpha=0.2)
ax3.legend(handles=[taking_off_patch, cruising_patch, underwater_patch],
           fontsize='large',
           loc='upper right',
           frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

plt.savefig(f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}_Test_Result.svg")
plt.savefig(f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}_Test_Result.png", dpi=1200)
# plt.savefig(f"../../Figures/EMS_MARL_Result_color.svg")
print(f"Total Reward: {ep_r:.2f}")
print(f"Test total time: {time_finish - time_start:.4f}s")
if step > 0:
    print(f"Average step cost: {(time_finish - time_start) / step * 1000:.4f} ms/step")
else:
    print("No steps were executed.")
print(f"Final Battery SOC: {soc_bat[-1] if soc_bat else 'N/A'}")
print(f"Final SuperCap SOC: {soc_sc_list[-1] if soc_sc_list else 'N/A'}")
plt.show()