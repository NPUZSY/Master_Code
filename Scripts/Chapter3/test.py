import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import matplotlib.patches as mpatches
import os
import json  # 新增：导入json模块
import argparse  # 新增：导入参数解析模块
from json import JSONEncoder  # 新增：导入JSON编码器基类

# 导入公共模块（与训练代码保持一致的导入形式）
from MARL_Engine import setup_project_root, device, IndependentDQN
project_root = setup_project_root()
from Scripts.Env import Envs
from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# ====================== 新增：自定义JSON编码器（处理numpy类型） ======================
class NumpyEncoder(JSONEncoder):
    """自定义JSON编码器，处理numpy类型和其他非标准类型"""
    def default(self, obj):
        # 处理numpy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理torch张量
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        # 处理其他数值类型
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        # 调用父类默认方法处理其他类型
        return super(NumpyEncoder, self).default(obj)

# ====================== 命令行参数解析 ======================
def parse_args():
    """解析命令行参数（指定待测试模型路径）"""
    parser = argparse.ArgumentParser(description='MARL模型测试脚本（支持指定待测试模型路径）')
    
    # 核心：模型路径参数（必选/可选）
    parser.add_argument('--net-date', type=str, required=True,
                        help='模型所在的日期文件夹（必填，如：1213）')
    parser.add_argument('--train-id', type=str, required=True,
                        help='模型对应的训练ID（必填，如：11）')
    
    # 可选配置参数
    parser.add_argument('--model-prefix', type=str, default="MARL_Model", help='模型前缀')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')
    parser.add_argument('--max-time', type=float, default=800.0, help='最大测试时长（秒，默认：800）')
    parser.add_argument('--sc-threshold', type=float, default=1e-3, help='超级电容非活跃阈值（默认：1e-3）')
    parser.add_argument('--show-plot', action='store_true', help='是否显示测试结果图（默认：仅保存不显示）')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录（默认：模型所在目录）')
    
    return parser.parse_args()

# 解析参数
args = parse_args()
# =====================================================================

# 全局设置（从命令行参数读取）
torch.manual_seed(args.seed)

# 环境参数（从环境实例中动态获取，而非硬编码）
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 40
N_SC_ACTIONS = 2

if __name__ == '__main__':
    # ====================== 动态配置模型路径（从命令行参数读取） ======================
    # 打印配置确认信息
    print("=" * 80)
    print("                    测试配置确认                  ")
    print("=" * 80)
    print(f"待测试模型路径:")
    print(f"  - 日期文件夹: {args.net_date}")
    print(f"  - 训练ID: {args.train_id}")
    print(f"  - 模型前缀: {args.model_prefix}")
    print(f"测试配置:")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 最大测试时长: {args.max_time}秒")
    print(f"  - 超级电容非活跃阈值: {args.sc_threshold}")
    print(f"  - 显示结果图: {'是' if args.show_plot else '否'}")
    print("=" * 80 + "\n")

    # 初始化环境
    env = Envs()
    
    # 动态获取状态维度（与训练代码保持一致）
    N_STATES = env.observation_space.shape[0]
    print(f"自动识别环境状态维度: N_STATES = {N_STATES}")

    # 初始化智能体（与训练代码参数保持一致）
    FC_Agent = IndependentDQN("FC_Agent", N_STATES, N_FC_ACTIONS)
    Bat_Agent = IndependentDQN("Bat_Agent", N_STATES, N_BAT_ACTIONS)
    SC_Agent = IndependentDQN("SC_Agent", N_STATES, N_SC_ACTIONS)

    # 构建模型路径（基于项目根路径 + 命令行参数）
    MODEL_BASE_DIR = os.path.join(project_root, "nets", "Chap3", args.net_date, args.train_id)
    # 自定义保存目录（优先使用命令行指定的，否则用模型目录）
    SAVE_DIR = args.save_dir if args.save_dir else MODEL_BASE_DIR
    MODEL_FILE_PREFIX = os.path.join(MODEL_BASE_DIR, args.model_prefix)
    
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
    # 新增：存储未匹配功率的列表
    unmatched_power_list = []
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
    # 新增：总未匹配功率初始化
    total_unmatched_power = 0.0
    total_unmatched_energy = 0.0  # 未匹配能量（功率×时间）
    episode_times = {
        'Action_Select': 0.0,
        'Env_Step': 0.0,
        'Logging_Processing': 0.0,
        'Other_Overhead': 0.0
    }
    sc_inactive_threshold = args.sc_threshold  # 从命令行参数读取
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
        if step > 500:
            pass
        s_, r, done, info = env.step(action_list)
        env_time = time.time() - t_env0

        # 统计数据收集
        total_fc_H2_g += float(info.get("C_fc_g", 0.0))
        total_bat_H2_g += float(info.get("C_bat_g", 0.0))
        times.append(step * dt)  # 修复2：时间轴基于dt，与Power_Profile对齐
        current_fc = float(s_[2])
        current_bat = float(s_[3])
        current_sc = float(s_[4])
        power_fc.append(current_fc)
        battery_power.append(current_bat)
        power_sc.append(current_sc)
        soc_bat.append(float(s_[5]))
        soc_sc_list.append(float(s_[6]))

        # 新增：计算当前步未匹配功率（负载需求 - 所有电源输出）
        # 负载需求：loads[step]（当前步的负载功率）
        # 总输出功率：燃料电池 + 电池 + 超级电容（注意符号：电池/电容放电为正，充电为负）
        if step < len(loads):
            load_demand = loads[step]
            total_supply = current_fc + current_bat + current_sc
            unmatched_power = load_demand - total_supply
            unmatched_power_list.append(unmatched_power)
            # 累加总未匹配功率（绝对值，代表供需失衡的总量）
            total_unmatched_power += abs(unmatched_power)
            # 累加未匹配能量（Wh）：|功率| × 时间步长（小时）
            total_unmatched_energy += abs(unmatched_power) * dt / 3600.0

        # 超级电容统计
        p_sc = current_sc
        if p_sc > 0:
            sc_release_power_sum += p_sc * dt
        elif p_sc < 0:
            sc_absorb_power_sum += (-p_sc) * dt

        # 电池充电统计
        if current_bat < 0:
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
        # 修复3：终止条件适配命令行指定的最大时长
        if done or step * dt >= args.max_time - dt:  # 从命令行参数读取最大时长
            break
        s = s_
        step += 1

    # 新增：计算未匹配功率相关统计
    avg_unmatched_power = total_unmatched_power / total_steps if total_steps > 0 else 0.0
    max_unmatched_power = max([abs(p) for p in unmatched_power_list]) if unmatched_power_list else 0.0
    # 未匹配功率占总负载需求的比例
    total_load_demand = sum([abs(loads[i]) for i in range(min(total_steps, len(loads)))])
    unmatched_ratio = (total_unmatched_power / total_load_demand * 100) if total_load_demand > 0 else 0.0

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

    # ====================== 整理测试结果为JSON格式 ======================
    test_results = {
        # 基础配置信息
        "config": {
            "model_info": {
                "net_date": args.net_date,
                "train_id": args.train_id,
                "model_prefix": args.model_prefix,
                "model_path": MODEL_FILE_PREFIX
            },
            "test_params": {
                "seed": args.seed,
                "max_time": args.max_time,
                "sc_threshold": args.sc_threshold,
                "dt": dt,
                "show_plot": args.show_plot,
                "save_dir": SAVE_DIR
            },
            "env_params": {
                "n_states": N_STATES,
                "n_fc_actions": N_FC_ACTIONS,
                "n_bat_actions": N_BAT_ACTIONS,
                "n_sc_actions": N_SC_ACTIONS
            }
        },
        # 时间统计
        "time_metrics": {
            "total_test_time_s": round(float(total_time), 4),
            "average_step_time_s": round(float(total_time / total_steps if total_steps > 0 else 0), 6),
            "total_steps": total_steps,
            "phase_time_breakdown_s": {
                "Action_Select": round(float(episode_times['Action_Select']), 4),
                "Env_Step": round(float(episode_times['Env_Step']), 4),
                "Logging_Processing": round(float(episode_times['Logging_Processing']), 4),
                "Other_Overhead": round(float(episode_times['Other_Overhead']), 4)
            }
        },
        # 氢耗统计
        "hydrogen_consumption": {
            "total_h2_g": round(float(total_h2), 6),
            "fc_h2_g": round(float(total_fc_H2_g), 6),
            "bat_h2_g": round(float(total_bat_H2_g), 6),
            "fc_h2_ratio": round(float(fc_h2_ratio * 100), 2),
            "bat_h2_ratio": round(float(bat_h2_ratio * 100), 2)
        },
        # 电池统计
        "battery_stats": {
            "soc_min": round(float(min(soc_bat) if soc_bat else 0), 6),
            "soc_max": round(float(max(soc_bat) if soc_bat else 0), 6),
            "soc_range": round(float(soc_bat_range), 6),
            "charge_steps": bat_charge_steps,
            "charge_time_s": round(float(bat_charge_steps * dt), 2),
            "charge_ratio": round(float(bat_charge_ratio * 100), 2)
        },
        # 超级电容统计
        "supercap_stats": {
            "release_energy_wh": round(float(sc_release_Wh), 6),
            "absorb_energy_wh": round(float(sc_absorb_Wh), 6),
            "inactive_steps": sc_inactive_steps,
            "inactive_ratio": round(float(sc_inactive_ratio * 100), 2)
        },
        # 功率匹配统计
        "power_matching": {
            "total_unmatched_power_w_step": round(float(total_unmatched_power), 6),
            "average_unmatched_power_w": round(float(avg_unmatched_power), 6),
            "max_unmatched_power_w": round(float(max_unmatched_power), 6),
            "total_unmatched_energy_wh": round(float(total_unmatched_energy), 6),
            "unmatched_ratio_percent": round(float(unmatched_ratio), 2),
            "total_load_demand_w_step": round(float(total_load_demand), 6)
        },
        # 核心性能指标
        "core_metrics": {
            "total_reward": round(float(ep_r), 2),
            "test_completed": True,
            "early_stop": done
        },
        # 原始数据（可选存储，便于后续分析）
        "raw_data": {
            "times": [round(float(t), 2) for t in times],
            "power_fc": [round(float(p), 2) for p in power_fc],
            "battery_power": [round(float(p), 2) for p in battery_power],
            "power_sc": [round(float(p), 2) for p in power_sc],
            "soc_bat": [round(float(s), 6) for s in soc_bat],
            "soc_sc": [round(float(s), 6) for s in soc_sc_list],
            "unmatched_power": [round(float(p), 2) for p in unmatched_power_list],
            "loads": [round(float(l), 2) for l in loads[:len(power_fc)]],
            "temperature": [round(float(t), 2) for t in temperature[:len(power_fc)]]
        },
        # 测试时间戳
        "timestamp": {
            "test_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_start)),
            "test_end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    }

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

    # 功率曲线（适配命令行指定的最大时长）
    l1, = ax1.plot(plot_times, plot_loads, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
    l2, = ax1.plot(plot_times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
    l3, = ax1.plot(plot_times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
    l6, = ax1.plot(plot_times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax1.set_xlim(0, args.max_time)  # 从命令行参数读取最大时长
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

    # 修复5：阶段背景匹配Power_Profile的时间分段（适配最大时长）
    phase_split = args.max_time / 4  # 均分4个阶段
    ax1.axvspan(0, phase_split, alpha=0.2, color='lightblue', label='Flight Phase')       # 飞行阶段
    ax1.axvspan(phase_split, 2*phase_split, alpha=0.2, color='lightgreen', label='Surface Sliding') # 水面滑行
    ax1.axvspan(2*phase_split, 3*phase_split, alpha=0.2, color='salmon', label='Underwater Navigation') # 水下潜航
    ax1.axvspan(3*phase_split, args.max_time, alpha=0.2, color='mediumpurple', label='Re-water Exit') # 再出水飞行

    # 图例配置（优化布局）
    lines = [l1, l2, l3, l6, l4, l7, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=LABEL_FONT_SIZE-2)
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 保存图像（使用命令行指定的保存目录）
    save_path_svg = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Result.svg")
    save_path_png = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Result.png")
    
    plt.savefig(save_path_svg, bbox_inches='tight', dpi=1200)
    plt.savefig(save_path_png, dpi=1200, bbox_inches='tight')
    
    print(f"\n📊 测试结果图已保存:")
    print(f"   SVG: {save_path_svg}")
    print(f"   PNG: {save_path_png}")

    # ====================== 保存JSON格式测试结果（使用自定义编码器） ======================
    json_save_path = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Results.json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        # 使用自定义编码器处理numpy类型
        json.dump(test_results, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    
    print(f"\n📄 JSON格式测试结果已保存:")
    print(f"   JSON: {json_save_path}")

    # 打印详细结果汇总
    print("\n" + "="*80)
    print("📈 测试结果汇总与分析")
    print("="*80)
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
    print(f"\n【功率匹配性能】")  # 新增：未匹配功率统计
    print(f"  总未匹配功率（绝对值累加）：{total_unmatched_power:.6f} W·步")
    print(f"  平均未匹配功率：{avg_unmatched_power:.6f} W/步")
    print(f"  最大单次未匹配功率：{max_unmatched_power:.6f} W")
    print(f"  总未匹配能量：{total_unmatched_energy:.6f} Wh")
    print(f"  未匹配功率占总负载比例：{unmatched_ratio:.2f}%")
    print(f"\n【性能指标】")
    print(f"  累积奖励：{ep_r:.2f}")
    print(f"  总测试步数：{total_steps} 步")
    print(f"  总耗时：{total_time:.4f}s")
    print(f"  平均步耗时：{total_time/total_steps:.6f}s/步")
    print("="*80)

    # 显示图像（根据命令行参数控制）
    if args.show_plot:
        plt.show()
    else:
        plt.close()  # 关闭图像释放内存
    
    print(f"\n✅ 测试完成！所有结果已保存至：{SAVE_DIR}")
    print(f"   📄 JSON结果文件：{json_save_path}")
    print(f"   📊 可视化文件：{save_path_svg} / {save_path_png}")