import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.patches as mpatches
import os

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
# 论文中使用的训练ID
train_id = '1'
net_name_base = 'bs64_lr10_ep_464_pool10_freq10_MARL_MARL_IQL_32x20x2_MAX_R-679'

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
bat_charge_time_s = bat_charge_steps * dt
bat_charge_ratio = bat_charge_steps / total_steps

# 3) 电池 SOC 全局变化范围
soc_bat_range = (max(soc_bat) - min(soc_bat)) if soc_bat else 0.0

# 4) 超级电容吸收/释放能量（J 与 Wh），以及功率匹配度（SC 未参与比例）
sc_absorb_J = sc_absorb_power_sum  # J (since power*dt with dt in s)
sc_release_J = sc_release_power_sum
sc_absorb_Wh = sc_absorb_J / 3600.0
sc_release_Wh = sc_release_J / 3600.0

# SC 未参与步数（|P_sc| < threshold）
sc_inactive_steps = sum(1 for p in power_sc if abs(p) < sc_inactive_threshold)
sc_inactive_ratio = sc_inactive_steps / total_steps  # 不依靠 SC 的工作点比例

# 5) RL 计时细分统计（总时长、平均每步、占比）
timing_summary = {}
for k, v in episode_times.items():
    timing_summary[k] = {
        'total_s': v,
        'avg_per_step_s': v / total_steps,
        'ratio': v / total_time if total_time > 0 else 0.0
    }

# --------------------------------------------------------------------
# 绘图（保持你原来的图；我没改布局，仅确保绘制使用最新数据）
# --------------------------------------------------------------------
# 定义颜色列表
best_color = ['#3570a8', '#f09639', '#42985e', '#c84343', '#8a7ab5']
article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
colors = article_color
LINES_ALPHA = 1
LABEL_FONT_SIZE = 18

fig, ax1 = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875, hspace=0.2, wspace=0.2)

l1, = ax1.plot(times, loads[:len(times)], label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
l2, = ax1.plot(times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
l3, = ax1.plot(times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
l6, = ax1.plot(times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
ax1.set_ylabel('Power/W', color='black', fontsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='x', labelcolor='black', labelsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='y', labelcolor='black', labelsize=LABEL_FONT_SIZE)

ax2 = ax1.twinx()
l4, = ax2.plot(times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
l7, = ax2.plot(times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)

ax2.set_ylabel('SOC', color='black', fontsize=LABEL_FONT_SIZE)
ax2.tick_params(axis='y', labelcolor='black', labelsize=LABEL_FONT_SIZE)

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 65))
l5, = ax3.plot(times, temperature[:len(times)], label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)

ax3.set_ylabel('Environment Temperature/℃', color=colors[4], fontsize=LABEL_FONT_SIZE)
ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)

lines = [l1, l2, l3, l6, l4, l7, l5]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower center', ncol=3)

plt.xlim(0, 600)
ax1.axvspan(0, 150, alpha=0.2, color='lightblue', label='Taking off & Climbing')
ax1.axvspan(150, 450, alpha=0.2, color='lightgreen', label='Cruising')
ax1.axvspan(450, 600, alpha=0.2, color='salmon', label='Descending & underwater')
ax1.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

taking_off_patch = mpatches.Patch(color='lightblue', label='Air flight', alpha=0.2)
cruising_patch = mpatches.Patch(color='lightgreen', label='Surface navigation', alpha=0.2)
underwater_patch = mpatches.Patch(color='salmon', label='Underwater navigation', alpha=0.2)
ax3.legend(handles=[taking_off_patch, cruising_patch, underwater_patch],
           fontsize='large',
           loc='upper right',
           frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

os.makedirs(f"../../nets/Chap3/{net_data}/{train_id}", exist_ok=True)
plt.savefig(f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}_Test_Result.svg")
plt.savefig(f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}_Test_Result.png", dpi=1200)

# --------------------------------------------------------------------
# 最终打印汇总信息（包含你要求的所有指标）
# --------------------------------------------------------------------
print("\n===================== 测试结果汇总与分析 =====================")

print(f"【等效氢耗】")
print(f"系统总等效氢耗：{total_h2:.6f} g")
print(f"  ├─ 燃料电池氢耗：{fc_h2:.6f} g，占比 {fc_h2_ratio*100:.2f}%")
print(f"  └─ 电池等效氢耗：{bat_h2:.6f} g，占比 {bat_h2_ratio*100:.2f}%")
print()

print(f"【电池 SOC 情况】")
print(f"电池 SOC 最低值：{min(soc_bat):.6f}")
print(f"电池 SOC 最高值：{max(soc_bat):.6f}")
print(f"电池 SOC 全局变化范围（最大-最小）：{soc_bat_range:.6f}")
print()

print(f"【电池充电（能量回收）特性】")
print(f"电池进入充电状态的时间：{bat_charge_time_s:.2f} s")
print(f"充电步数比例：{bat_charge_ratio*100:.2f}%")
print()

print(f"【超级电容能量流动特性】")
print(f"超级电容释放能量：{sc_release_J:.4f} J  (约 {sc_release_Wh:.6f} Wh)")
print(f"超级电容吸收能量：{sc_absorb_J:.4f} J  (约 {sc_absorb_Wh:.6f} Wh)")
print(f"SC 未参与工作（|P_sc| < 阈值）的步数：{sc_inactive_steps}/{total_steps}  ({sc_inactive_ratio*100:.2f}%)")
print(f"功率匹配度（不依赖超级电容的比例）：{sc_inactive_ratio*100:.2f}%")
print()

print(f"【整体回报与时间性能】")
print(f"智能体累积奖励：{ep_r:.2f}")
print(f"测试总耗时：{total_time:.4f} s")
if total_steps > 0:
    print(f"平均单步耗时：{total_time/total_steps*1000:.4f} ms/step")
print()

print("【动作选择、环境步进、日志处理等阶段的详细耗时占比】")
print("阶段名称                     总耗时(s)      平均每步耗时(ms)    占测试总时长(%)")
for k, v in timing_summary.items():
    print(f"{k:20s}   {v['total_s']:.6f} s      {(v['avg_per_step_s']*1000):.6f} ms/step      {(v['ratio']*100):.2f}%")

print("==============================================================")


plt.show()
