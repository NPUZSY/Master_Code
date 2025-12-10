import matplotlib.pyplot as plt
import torch
import time
import matplotlib.patches as mpatches

from train import DQN
from Env_FC_Li import Envs
torch.manual_seed(0)
env = Envs()
dqn = DQN()
net_data = '0605'
train_id = '14'
net_name = 'bs64_lr50_episode_1000_pool10_freq1_basemodel__little_fc_power'
dqn.load_net(f"../nets/{net_data}/{train_id}/{net_name}")
s = env.reset()
step = 0
power_fc = []
battery_power = []
times = []
soc = []
loads = env.loads[:-1]
temperature = env.temperature[:-1]
time_start = time.time()
ep_r = 0  # 初始化ep_r
while True:
    a = dqn.choose_action(s, train=False)

    # take action
    s_, r, done, _ = env.step(a)

    # print(f"s: {s} s_:{s_} r: {r}")
    times.append(step)

    power_fc.append(float(env.power_fc))
    battery_power.append(s[3])
    soc.append(float(env.battery.soc))
    # print(r)
    # print(f"Step: {step} Loads: {env.loads[step]} Action: {a} FC_Power: {env.power_fc} Reward: {r} State: {s}")

    # dqn.store_transition(s, a, r, s_)

    ep_r += r

    if done:
        break
    s = s_
    step += 1
    if step > 500:
        pass
time_finish = time.time()
# 定义颜色列表
best_color = ['#3570a8', '#f09639', '#42985e', '#c84343', '#8a7ab5']
article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
colors = article_color
LINES_ALPHA = 1
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

l1, = ax1.plot(times, loads, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
l2, = ax1.plot(times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
l3, = ax1.plot(times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)

LABEL_FONT_SIZE = 18

# 设置主坐标轴的标签和颜色
ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
ax1.set_ylabel('Power/W', color='black', fontsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='x', labelcolor='black', labelsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='y', labelcolor='black', labelsize=LABEL_FONT_SIZE)

# 创建第二个 y 轴（共享 x 轴）
ax2 = ax1.twinx()
# ax2.spines['right'].set_position(('outward', 20))
l4, = ax2.plot(times, soc, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)

# 设置第二个 y 轴的标签和颜色
ax2.set_ylabel('Battery SOC', color=colors[3], fontsize=LABEL_FONT_SIZE)
ax2.tick_params(axis='y', labelcolor=colors[3], labelsize=LABEL_FONT_SIZE)


ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 65))
l5, = ax3.plot(times, temperature, label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)

# 设置第二个 y 轴的标签和颜色
ax3.set_ylabel('Environment Temperature/°C', color=colors[4], fontsize=LABEL_FONT_SIZE)
ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)

# 合并两个坐标轴的图例
lines = [l1, l2, l3, l4, l5]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc='lower center')

# 设置横轴和纵轴显示范围
plt.xlim(0, 600)
plt.ylim(30, -100)   # 温度范围

# 起飞阶段背景和数据
ax1.axvspan(0, 150, alpha=0.2, color='lightblue', label='Taking off & Climbing')
# 巡航阶段背景和数据
ax1.axvspan(150, 450, alpha=0.2, color='lightgreen', label='Cruising')
# 降落阶段背景和数据
ax1.axvspan(450, 600, alpha=0.2, color='salmon', label='Descending & underwater')

# 添加网格线
ax1.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

taking_off_patch = mpatches.Patch(color='lightblue', label='Air flight', alpha=0.2)
cruising_patch = mpatches.Patch(color='lightgreen', label='Surface navigation', alpha=0.2)
underwater_patch = mpatches.Patch(color='salmon', label='Underwater navigation', alpha=0.2)
ax3.legend(handles=[taking_off_patch, cruising_patch, underwater_patch],
                fontsize='large',
                loc='upper right',
                frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

plt.savefig(f"../nets/{net_data}/{train_id}/{net_name}.svg")
plt.savefig(f"../Figures/EMS_Result_result_color.svg")
print(f"Total Reward:{ep_r}")
plt.show()
print(f"Test total time:{time_finish - time_start}s Step cost{(time_finish - time_start) / 600}s")
print(f"SOC Result:{soc}")




