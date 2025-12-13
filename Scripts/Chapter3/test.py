import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import matplotlib.patches as mpatches
import os

# 导入公共模块（与训练代码保持一致的导入形式）
from MARL_Engine import setup_project_root, device, IndependentDQN
project_root = setup_project_root()
from Scripts.Env import Envs
from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# 全局设置
torch.manual_seed(0)

# 环境参数（从环境实例中动态获取，而非硬编码）
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2

if __name__ == '__main__':
    # 初始化环境
    env = Envs()
    
    # 动态获取状态维度（与训练代码保持一致）
    N_STATES = env.observation_space.shape[0]
    print(f"自动识别环境状态维度: N_STATES = {N_STATES}")

    # 模型路径配置（使用项目根路径拼接，支持任意路径执行）
    net_data = '1213'          # 日期文件夹
    train_id = '3'             # 训练ID
    net_name_base = 'bs64_lr10_ep_173_pool100_freq10_MARL_MARL_IQL_32x20x2_MAX_R-57'

    # 初始化智能体（与训练代码参数保持一致）
    FC_Agent = IndependentDQN("FC_Agent", N_STATES, N_FC_ACTIONS)
    Bat_Agent = IndependentDQN("Bat_Agent", N_STATES, N_BAT_ACTIONS)
    SC_Agent = IndependentDQN("SC_Agent", N_STATES, N_SC_ACTIONS)

    # 构建模型路径（基于项目根路径，支持任意工作目录执行）
    MODEL_BASE_DIR = os.path.join(project_root, "nets", "Chap3", net_data, train_id)
    MODEL_FILE_PREFIX = os.path.join(MODEL_BASE_DIR, net_name_base)
    
    # 加载模型（增加路径合法性检查）
    try:
        # 确保模型目录存在
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        
        # 加载各智能体模型
        FC_Agent.load_net(f"{MODEL_FILE_PREFIX}_FC.pth")
        Bat_Agent.load_net(f"{MODEL_FILE_PREFIX}_BAT.pth")
        SC_Agent.load_net(f"{MODEL_FILE_PREFIX}_SC.pth")
        
        print(f"✅ 成功加载MARL模型:")
        print(f"   模型路径: {MODEL_FILE_PREFIX}_*.pth")
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        print(f"   期望路径: {MODEL_FILE_PREFIX}_*.pth")
        raise
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
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
    # 修复1：统一数据长度（去掉[:-1]避免维度不匹配）
    loads = env.loads
    temperature = env.temperature
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
    print("\n🚀 开始测试...")
    while True:
        t0_loop = time.time()

        # 动作选择
        t_as0 = time.time()
        a_fc = FC_Agent.choose_action(s, train=False)
        a_bat = Bat_Agent.choose_action(s, train=False)
        a_sc = SC_Agent.choose_action(s, train=False)
        action_list = [a_fc, a_bat, a_sc]
        action_time = time.time() - t_as0

        # 环境交互
        t_env0 = time.time()
        s_, r, done, info = env.step(action_list)
        env_time = time.time() - t_env0

        # 统计数据收集
        total_fc_H2_g += float(info.get("C_fc_g", 0.0))
        total_bat_H2_g += float(info.get("C_bat_g", 0.0))
        times.append(step * dt)  # 修复2：时间轴基于dt，与Power_Profile对齐
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
        log_time = time.time() - t_log0

        # 累加各阶段总耗时
        episode_times['Action_Select'] += action_time
        episode_times['Env_Step'] += env_time
        episode_times['Logging_Processing'] += log_time

        # 计算其他开销
        t1_loop = time.time()
        loop_time = t1_loop - t0_loop
        episode_times['Other_Overhead'] += loop_time - (action_time + env_time + log_time)

        total_steps += 1
        # 修复3：终止条件适配800秒总时长
        if done or step * dt >= 800 - dt:  # 匹配Power_Profile的800秒总时长
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

    # 绘图配置（适配Power_Profile的最新修改）
    plt.rcParams.update({
        'font.family': ['Times New Roman'],  # 兼容中英文
        'axes.unicode_minus': False,
        'font.size': 12
    })
    best_color = ['#3570a8', '#f09639', '#42985e', '#c84343', '#8a7ab5']
    article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    colors = article_color
    LINES_ALPHA = 1
    LABEL_FONT_SIZE = 18

    # 绘制结果图
    fig, ax1 = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875)
    
    # 修复4：统一数据长度（截断到实际测试步数）
    plot_times = times[:len(power_fc)]
    plot_loads = loads[:len(power_fc)]
    plot_temperature = temperature[:len(power_fc)]

    # 功率曲线（适配800秒时长）
    l1, = ax1.plot(plot_times, plot_loads, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
    l2, = ax1.plot(plot_times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
    l3, = ax1.plot(plot_times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
    l6, = ax1.plot(plot_times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax1.set_xlim(0, 800)  # 匹配Power_Profile的800秒总时长
    ax1.set_ylim(-2500, 5500)  # 匹配功率峰值5000W

    # SOC曲线
    ax2 = ax1.twinx()
    l4, = ax2.plot(plot_times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
    l7, = ax2.plot(plot_times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)
    ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)
    ax2.set_ylim(0, 1.0)  # SOC范围0-1

    # 温度曲线（适配-25~40℃范围）
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 65))
    l5, = ax3.plot(plot_times, plot_temperature, label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)
    ax3.set_ylabel('Environment Temperature/°C', color=colors[4], fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)
    ax3.set_ylim(-25, 40)  # 匹配Power_Profile的温度轴范围

    # 修复5：阶段背景匹配Power_Profile的时间分段
    ax1.axvspan(0, 200, alpha=0.2, color='lightblue', label='Flight Phase')       # 飞行阶段
    ax1.axvspan(200, 400, alpha=0.2, color='lightgreen', label='Surface Sliding') # 水面滑行
    ax1.axvspan(400, 600, alpha=0.2, color='salmon', label='Underwater Navigation') # 水下潜航
    ax1.axvspan(600, 800, alpha=0.2, color='mediumpurple', label='Re-water Exit') # 再出水飞行

    # 图例配置（优化布局）
    lines = [l1, l2, l3, l6, l4, l7, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=LABEL_FONT_SIZE-2)
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    # 保存图像（使用项目根路径，确保保存路径正确）
    save_path_svg = os.path.join(MODEL_BASE_DIR, f"{net_name_base}_Test_Result.svg")
    save_path_png = os.path.join(MODEL_BASE_DIR, f"{net_name_base}_Test_Result.png")
    
    plt.savefig(save_path_svg, bbox_inches='tight', dpi=1200)
    plt.savefig(save_path_png, dpi=1200, bbox_inches='tight')
    
    print(f"\n📊 测试结果图已保存:")
    print(f"   SVG: {save_path_svg}")
    print(f"   PNG: {save_path_png}")

    # 打印详细结果汇总
    print("\n" + "="*60)
    print("📈 测试结果汇总与分析")
    print("="*60)
    print(f"【等效氢耗】")
    print(f"  系统总等效氢耗：{total_h2:.6f} g")
    print(f"  ├─ 燃料电池氢耗：{total_fc_H2_g:.6f} g（{fc_h2_ratio*100:.2f}%）")
    print(f"  └─ 电池等效氢耗：{total_bat_H2_g:.6f} g（{bat_h2_ratio*100:.2f}%）")
    print(f"\n【电池 SOC 情况】")
    print(f"  电池 SOC 范围：{min(soc_bat):.6f} ~ {max(soc_bat):.6f}")
    print(f"  电池 SOC 变化幅度：{soc_bat_range:.6f}")
    print(f"\n【电池充电特性】")
    print(f"  充电步数：{bat_charge_steps} 步（{bat_charge_steps*dt:.2f}s）")
    print(f"  充电占比：{bat_charge_ratio*100:.2f}%")
    print(f"\n【超级电容特性】")
    print(f"  释放能量：{sc_release_Wh:.6f} Wh")
    print(f"  吸收能量：{sc_absorb_Wh:.6f} Wh")
    print(f"  未参与比例：{sc_inactive_ratio*100:.2f}%")
    print(f"\n【性能指标】")
    print(f"  累积奖励：{ep_r:.2f}")
    print(f"  总测试步数：{total_steps} 步")
    print(f"  总耗时：{total_time:.4f}s")
    print(f"  平均步耗时：{total_time/total_steps:.6f}s/步")
    print("="*60)

    # 可选：显示图像
    # plt.show()
    
    print(f"\n✅ 测试完成！所有结果已保存至：{MODEL_BASE_DIR}")