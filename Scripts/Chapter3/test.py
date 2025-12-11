import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import matplotlib.patches as mpatches
import os

# 导入公共模块
from MARL_Engine import setup_project_root, device, IndependentDQN, font_get
project_root = setup_project_root()
from Scripts.Env import Envs

# 获取新罗马
font_get()

# 环境参数
N_STATES = 7  # [load, temp, P_fc, P_bat, P_sc, SOC_b, SOC_sc]
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2
torch.manual_seed(0)

if __name__ == '__main__':
    # 初始化环境
    env = Envs()

    # 模型路径配置
    net_data = '1210'
    train_id = '12'
    net_name_base = 'bs32_lr20_ep_105_pool10_freq10_MARL_MARL_IQL_32x20x2_MAX_R-3502'

    # 初始化智能体
    FC_Agent = IndependentDQN("FC_Agent", N_STATES, N_FC_ACTIONS)
    Bat_Agent = IndependentDQN("Bat_Agent", N_STATES, N_BAT_ACTIONS)
    SC_Agent = IndependentDQN("SC_Agent", N_STATES, N_SC_ACTIONS)

    # 加载模型
    BASE_PATH = f"../../nets/Chap3/{net_data}/{train_id}/{net_name_base}"
    try:
        FC_Agent.load_net(f"{BASE_PATH}_FC.pth")
        Bat_Agent.load_net(f"{BASE_PATH}_BAT.pth")
        SC_Agent.load_net(f"{BASE_PATH}_SC.pth")
        print(f"Successfully loaded MARL models from: {BASE_PATH}_*.pth")
    except FileNotFoundError:
        print(f"Error: Model files not found in {BASE_PATH}_*.pth")
        raise

    # 测试初始化
    s = env.reset()
    step = 0
    power_fc = []
    battery_power = []
    power_sc = []
    soc_bat = []
    soc_sc_list = []
    times = []
    loads = env.loads[:-1]
    temperature = env.temperature[:-1]
    ep_r = 0

    # 统计变量初始化
    total_fc_H2_g = 0.0
    total_bat_H2_g = 0.0
    sc_release_power_sum = 0.0
    sc_absorb_power_sum = 0.0
    bat_charge_steps = 0
    total_steps = 0
    episode_times = {
        'Action_Select': 0.0,
        'Env_Step': 0.0,
        'Logging_Processing': 0.0,
        'Other_Overhead': 0.0
    }
    sc_inactive_threshold = 1e-3
    dt = getattr(env, "dt", 1.0)
    time_start = time.time()

    # 测试主循环
    # 测试主循环
    while True:
        t0_loop = time.time()

        # 动作选择
        t_as0 = time.time()
        a_fc = FC_Agent.choose_action(s, train=False)
        a_bat = Bat_Agent.choose_action(s, train=False)
        a_sc = SC_Agent.choose_action(s, train=False)
        action_list = [a_fc, a_bat, a_sc]
        action_time = time.time() - t_as0  # 当前步动作选择耗时

        # 环境交互
        t_env0 = time.time()
        s_, r, done, info = env.step(action_list)
        env_time = time.time() - t_env0    # 当前步环境交互耗时

        # 统计数据
        total_fc_H2_g += float(info.get("C_fc_g", 0.0))
        total_bat_H2_g += float(info.get("C_bat_g", 0.0))
        times.append(step)
        power_fc.append(float(s_[2]))
        battery_power.append(float(s_[3]))
        power_sc.append(float(s_[4]))
        soc_bat.append(float(s_[5]))
        soc_sc_list.append(float(s_[6]))

        # 超级电容统计
        p_sc = float(s_[4])
        if p_sc > 0:
            sc_release_power_sum += p_sc * dt
        elif p_sc < 0:
            sc_absorb_power_sum += (-p_sc) * dt

        # 电池充电统计
        if float(s_[3]) < 0:
            bat_charge_steps += 1

        ep_r += r
        t_log0 = time.time()
        log_time = time.time() - t_log0  # 当前步日志处理耗时

        # 累加各阶段总耗时
        episode_times['Action_Select'] += action_time
        episode_times['Env_Step'] += env_time
        episode_times['Logging_Processing'] += log_time

        # 计算其他开销
        t1_loop = time.time()
        loop_time = t1_loop - t0_loop
        episode_times['Other_Overhead'] += loop_time - (action_time + env_time + log_time)

        total_steps += 1
        if done or step >= len(loads)-1:
            break
        s = s_
        step += 1

    # 结果计算
    total_time = time.time() - time_start
    total_h2 = total_fc_H2_g + total_bat_H2_g
    fc_h2_ratio = total_fc_H2_g / total_h2 if total_h2 > 0 else 0.0
    bat_h2_ratio = total_bat_H2_g / total_h2 if total_h2 > 0 else 0.0
    bat_charge_ratio = bat_charge_steps / total_steps if total_steps > 0 else 0.0
    soc_bat_range = max(soc_bat) - min(soc_bat) if soc_bat else 0.0
    sc_absorb_Wh = sc_absorb_power_sum / 3600.0
    sc_release_Wh = sc_release_power_sum / 3600.0
    sc_inactive_steps = sum(1 for p in power_sc if abs(p) < sc_inactive_threshold)
    sc_inactive_ratio = sc_inactive_steps / total_steps if total_steps > 0 else 0.0

    # 绘图配置
    plt.rcParams['font.family'] = 'Times New Roman'
    best_color = ['#3570a8', '#f09639', '#42985e', '#c84343', '#8a7ab5']
    article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    colors = article_color
    LINES_ALPHA = 1
    LABEL_FONT_SIZE = 18

    # 绘制结果图
    fig, ax1 = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875)
    l1, = ax1.plot(times, loads[:len(times)], label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
    l2, = ax1.plot(times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
    l3, = ax1.plot(times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
    l6, = ax1.plot(times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)

    ax2 = ax1.twinx()
    l4, = ax2.plot(times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
    l7, = ax2.plot(times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)
    ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 65))
    l5, = ax3.plot(times, temperature[:len(times)], label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)
    ax3.set_ylabel('Environment Temperature/℃', color=colors[4], fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)

    # 图例配置
    lines = [l1, l2, l3, l6, l4, l7, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower center', ncol=3)
    ax1.axvspan(0, 150, alpha=0.2, color='lightblue')
    ax1.axvspan(150, 450, alpha=0.2, color='lightgreen')
    ax1.axvspan(450, 600, alpha=0.2, color='salmon')
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    # 保存图像
    save_dir = f"../../nets/Chap3/{net_data}/{train_id}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{net_name_base}_Test_Result.svg")
    plt.savefig(f"{save_dir}/{net_name_base}_Test_Result.png", dpi=1200)

    # 打印结果汇总
    print("\n===================== 测试结果汇总与分析 =====================")
    print(f"【等效氢耗】")
    print(f"系统总等效氢耗：{total_h2:.6f} g")
    print(f"  ├─ 燃料电池氢耗：{total_fc_H2_g:.6f} g，占比 {fc_h2_ratio*100:.2f}%")
    print(f"  └─ 电池等效氢耗：{total_bat_H2_g:.6f} g，占比 {bat_h2_ratio*100:.2f}%")
    print(f"\n【电池 SOC 情况】")
    print(f"电池 SOC 范围：{min(soc_bat):.6f} - {max(soc_bat):.6f}，变化幅度：{soc_bat_range:.6f}")
    print(f"\n【电池充电特性】")
    print(f"充电时间：{bat_charge_steps*dt:.2f}s，占比 {bat_charge_ratio*100:.2f}%")
    print(f"\n【超级电容特性】")
    print(f"释放能量：{sc_release_Wh:.6f} Wh，吸收能量：{sc_absorb_Wh:.6f} Wh")
    print(f"未参与比例：{sc_inactive_ratio*100:.2f}%")
    print(f"\n【性能指标】")
    print(f"累积奖励：{ep_r:.2f}，总耗时：{total_time:.4f}s")
    print("==============================================================")

    # plt.show()