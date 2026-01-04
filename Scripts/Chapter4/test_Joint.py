import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import os
import json
import argparse
import sys
from json import JSONEncoder
import torch.nn as nn
import torch.nn.functional as F

# ====================== 1. 环境与路径配置 ======================
def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 Scripts/Chapter4/，向上两级到项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# 导入原有引擎组件
from Scripts.Chapter3.MARL_Engine import Net, IndependentDQN, device
from Scripts.utils.global_utils import font_get

# 支持超级环境
from Scripts.Chapter5.Env_Ultra import EnvUltra

# 获取字体设置
font_get()

# ====================== 2. JointNet 相关类定义 ======================
class MultiTaskRNN(nn.Module):
    """适配 7 维输入的多任务 RNN 结构"""
    def __init__(self, input_dim=7, hidden_dim_rnn=256, num_layers_rnn=2, hidden_dim_fc=64):
        super(MultiTaskRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim_rnn, num_layers=num_layers_rnn, batch_first=True)
        self.fc_rnn_to_64 = nn.Linear(hidden_dim_rnn, hidden_dim_fc)
        self.reg_head = nn.Linear(hidden_dim_fc, 1)
        self.cls_head = nn.Linear(hidden_dim_fc, 4)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn[:, -1, :]
        feature_64 = F.relu(self.fc_rnn_to_64(out_rnn))
        return self.reg_head(feature_64), self.cls_head(feature_64), feature_64

class JointNet(nn.Module):
    """拼接 RNN 特征(64) + 回归值(1) = 65维输入 MARL Head"""
    def __init__(self, rnn_part, marl_head):
        super(JointNet, self).__init__()
        self.rnn_part = rnn_part
        self.marl_part = marl_head

    def forward(self, x):
        reg_out, _, feature_64 = self.rnn_part(x)
        joint_input = torch.cat([feature_64, reg_out], dim=1)
        return self.marl_part(joint_input)

class JointDQN(IndependentDQN):
    """支持 7 维输入并自动执行内部拼接的智能体"""
    def __init__(self, agent_name, rnn_model, n_actions):
        super(JointDQN, self).__init__(agent_name, 65, n_actions)
        self.n_actions = n_actions
        self.eval_net = JointNet(rnn_model, self.eval_net).to(device)
        self.target_net = JointNet(rnn_model, self.target_net).to(device)

    def choose_action(self, x, train=False, epsilon=0.9):
        x_tensor = torch.FloatTensor(x).to(device)
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        
        if train and np.random.uniform() >= epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                actions_value = self.eval_net(x_tensor)
            return torch.max(actions_value, 1)[1].item()

# ====================== 3. 工具类与参数解析 ======================
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

def parse_args():
    """解析命令行参数（指定待测试模型路径）"""
    parser = argparse.ArgumentParser(description='JointNet模型测试脚本（支持超级环境）')
    
    # 核心：模型路径参数（必选/可选）
    parser.add_argument('--net-date', type=str, required=True,
                        help='模型所在的日期文件夹（必填，如：1213）')
    parser.add_argument('--train-id', type=str, required=True,
                        help='模型对应的训练ID（必填，如：11）')
    parser.add_argument('--rnn-path', type=str, 
                        default=os.path.join(project_root, "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth"),
                        help='预训练RNN模型路径')
    
    # 新增：超级环境参数
    parser.add_argument('--use-ultra-env', action='store_true',
                        help='是否使用超级环境（EnvUltra）')
    parser.add_argument('--scenario', type=str, default='default',
                        help='超级环境场景类型（如：cruise, recon, rescue等，default表示经典环境）')
    
    # 可选配置参数
    parser.add_argument('--model-prefix', type=str, default="Joint_Model", help='模型前缀')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')
    parser.add_argument('--max-time', type=float, default=800.0, help='最大测试时长（秒，默认：800）')
    parser.add_argument('--sc-threshold', type=float, default=1e-3, help='超级电容非活跃阈值（默认：1e-3）')
    parser.add_argument('--show-plot', action='store_true', help='是否显示测试结果图（默认：仅保存不显示）')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录（默认：模型所在目录）')
    
    # 测试示例脚本：
    # python Scripts/Chapter4/test_joint.py --net-date 1219 --train-id 5 --rnn-path "your_rnn_path.pth"
    
    return parser.parse_args()

# ====================== 4. 主测试程序 ======================
if __name__ == '__main__':
    args = parse_args()
    
    # 打印配置确认信息
    print("=" * 80)
    print("                    JointNet测试配置确认                  ")
    print("=" * 80)
    print(f"待测试模型路径:")
    print(f"  - 日期文件夹: {args.net_date}")
    print(f"  - 训练ID: {args.train_id}")
    print(f"  - 模型前缀: {args.model_prefix}")
    print(f"  - RNN模型路径: {args.rnn_path}")
    print(f"测试配置:")
    print(f"  - 随机种子: {args.seed}")
    print(f"  - 最大测试时长: {args.max_time}秒")
    print(f"  - 超级电容非活跃阈值: {args.sc_threshold}")
    print(f"  - 显示结果图: {'是' if args.show_plot else '否'}")
    print("=" * 80 + "\n")
    
    torch.manual_seed(args.seed)
    
    # 初始化环境
    if args.use_ultra_env:
        # 使用超级环境
        env = EnvUltra(scenario_type=args.scenario)
        print(f"✅ 使用超级环境 EnvUltra，场景: {args.scenario}")
    else:
        # 使用经典环境
        from Scripts.Env import Envs
        env = Envs()
        print(f"✅ 使用经典环境 Envs")
    
    dt = getattr(env, "dt", 1.0)
    loads = env.loads
    temperature = env.temperature
    
    # 加载 RNN
    try:
        rnn_model = MultiTaskRNN().to(device)
        rnn_model.load_state_dict(torch.load(args.rnn_path, map_location=device))
        rnn_model.eval()
        print(f"✅ 成功加载RNN模型: {args.rnn_path}")
    except FileNotFoundError as e:
        print(f"❌ RNN模型文件未找到: {e}")
        raise
    except Exception as e:
        print(f"❌ RNN模型加载失败: {e}")
        raise

    # 初始化智能体（与JointNet适配）
    N_FC_ACTIONS = 32
    N_BAT_ACTIONS = 40
    N_SC_ACTIONS = 2
    
    FC_Agent = JointDQN("FC_Agent", rnn_model, N_FC_ACTIONS)
    Bat_Agent = JointDQN("Bat_Agent", rnn_model, N_BAT_ACTIONS)
    SC_Agent = JointDQN("SC_Agent", rnn_model, N_SC_ACTIONS)

    # 路径设置
    MODEL_BASE_DIR = os.path.join(project_root, "nets", "Chap4", "Joint_Net", args.net_date, args.train_id)
    # 自定义保存目录（优先使用命令行指定的，否则用模型目录）
    SAVE_DIR = args.save_dir if args.save_dir else MODEL_BASE_DIR
    MODEL_FILE_PREFIX = os.path.join(MODEL_BASE_DIR, args.model_prefix)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载权重
    try:
        FC_Agent.load_net(f"{MODEL_FILE_PREFIX}_FC.pth")
        Bat_Agent.load_net(f"{MODEL_FILE_PREFIX}_BAT.pth")
        SC_Agent.load_net(f"{MODEL_FILE_PREFIX}_SC.pth")
        print(f"✅ 成功加载JointNet模型:")
        print(f"   模型路径: {MODEL_FILE_PREFIX}_*.pth")
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        print(f"   期望路径: {MODEL_FILE_PREFIX}_*.pth")
        raise
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    # --- 循环变量初始化 ---
    s = env.reset()
    step = 0
    power_fc, battery_power, power_sc = [], [], []
    soc_bat, soc_sc_list, times, unmatched_power_list = [], [], [], []
    ep_r, total_fc_H2_g, total_bat_H2_g = 0, 0.0, 0.0
    
    # 统计变量初始化
    sc_release_power_sum = 0.0
    sc_absorb_power_sum = 0.0
    sc_inactive_steps = 0
    bat_charge_steps = 0
    total_unmatched_power = 0.0
    total_unmatched_energy = 0.0
    total_steps = 0
    
    # 时间统计
    episode_times = {
        'Action_Select': 0.0,
        'Env_Step': 0.0,
        'Logging_Processing': 0.0,
        'Other_Overhead': 0.0
    }
    
    time_start = time.time()

    print("🚀 开始测试...")
    while True:
        t0_loop = time.time()

        # 动作选择 (7维输入)
        t_as0 = time.time()
        a_fc = FC_Agent.choose_action(s, train=False)
        a_bat = Bat_Agent.choose_action(s, train=False)
        a_sc = SC_Agent.choose_action(s, train=False)
        action_time = time.time() - t_as0

        # 环境交互
        t_env0 = time.time()
        s_, r, done, info = env.step([a_fc, a_bat, a_sc])
        env_time = time.time() - t_env0

        # 记录数据
        t_log0 = time.time()
        times.append(step * dt)
        cur_fc, cur_bat, cur_sc = float(s_[2]), float(s_[3]), float(s_[4])
        power_fc.append(cur_fc)
        battery_power.append(cur_bat)
        power_sc.append(cur_sc)
        soc_bat.append(float(s_[5]))
        soc_sc_list.append(float(s_[6]))

        # 未匹配功率计算
        if step < len(loads):
            load_demand = loads[step]
            total_supply = cur_fc + cur_bat + cur_sc
            unmatch = load_demand - total_supply
            unmatched_power_list.append(unmatch)
            total_unmatched_power += abs(unmatch)
            total_unmatched_energy += abs(unmatch) * dt / 3600.0

        # 超级电容统计
        if abs(cur_sc) < args.sc_threshold: 
            sc_inactive_steps += 1
        if cur_sc > 0:
            sc_release_power_sum += cur_sc * dt
        elif cur_sc < 0:
            sc_absorb_power_sum += (-cur_sc) * dt

        # 电池充电统计
        if cur_bat < 0: 
            bat_charge_steps += 1
        
        total_fc_H2_g += float(info.get("C_fc_g", 0.0))
        total_bat_H2_g += float(info.get("C_bat_g", 0.0))
        ep_r += r
        
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
        if done or step * dt >= args.max_time - dt: 
            break
        s = s_
        step += 1

    # 计算测试总耗时
    total_time_cost = time.time() - time_start

    # ====================== 结果计算 ======================
    # 未匹配功率相关统计
    avg_unmatched_power = total_unmatched_power / total_steps if total_steps > 0 else 0.0
    max_unmatched_power = max([abs(p) for p in unmatched_power_list]) if unmatched_power_list else 0.0
    total_load_demand = sum([abs(loads[i]) for i in range(min(total_steps, len(loads)))])
    unmatched_ratio = (total_unmatched_power / total_load_demand * 100) if total_load_demand > 0 else 0.0

    # 氢耗统计
    total_h2 = total_fc_H2_g + total_bat_H2_g
    fc_h2_ratio = total_fc_H2_g / total_h2 if total_h2 > 0 else 0.0
    bat_h2_ratio = total_bat_H2_g / total_h2 if total_h2 > 0 else 0.0

    # 电池统计
    soc_bat_min = min(soc_bat) if soc_bat else 0.0
    soc_bat_max = max(soc_bat) if soc_bat else 0.0
    soc_bat_range = soc_bat_max - soc_bat_min
    bat_charge_ratio = bat_charge_steps / total_steps if total_steps > 0 else 0.0

    # 超级电容统计
    sc_inactive_ratio = sc_inactive_steps / total_steps if total_steps > 0 else 0.0
    sc_absorb_Wh = sc_absorb_power_sum / 3600.0
    sc_release_Wh = sc_release_power_sum / 3600.0

    # 时间统计
    avg_step_time = total_time_cost / total_steps if total_steps > 0 else 0.0

    # ====================== 整理测试结果为JSON格式 ======================
    test_results = {
        # 基础配置信息
        "config": {
            "model_info": {
                "net_date": args.net_date,
                "train_id": args.train_id,
                "model_prefix": args.model_prefix,
                "rnn_path": args.rnn_path,
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
                "n_fc_actions": N_FC_ACTIONS,
                "n_bat_actions": N_BAT_ACTIONS,
                "n_sc_actions": N_SC_ACTIONS
            }
        },
        # 时间统计
        "time_metrics": {
            "total_test_time_s": round(float(total_time_cost), 4),
            "average_step_time_s": round(float(avg_step_time), 6),
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
            "soc_min": round(float(soc_bat_min), 6),
            "soc_max": round(float(soc_bat_max), 6),
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

    # ====================== 绘图部分 ======================
    # 绘图配置（适配Power_Profile的最新修改）
    plt.rcParams.update({
        'font.family': ['Times New Roman'],  # 仅使用新罗马字体
        'axes.unicode_minus': False,
        'font.size': 12
    })
    article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    colors = article_color
    LINES_ALPHA = 1
    LABEL_FONT_SIZE = 18

    # 统一数据长度（截断到实际测试步数）
    plot_times = times[:len(power_fc)]
    plot_loads = loads[:len(power_fc)]
    plot_temperature = temperature[:len(power_fc)]

    # --- 总图绘制 ---
    fig, ax1 = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875)
    
    # 功率曲线
    l1, = ax1.plot(plot_times, plot_loads, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
    l2, = ax1.plot(plot_times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
    l3, = ax1.plot(plot_times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
    l6, = ax1.plot(plot_times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax1.set_xlim(0, args.max_time)
    ax1.set_ylim(-2500, 5500)

    # SOC曲线
    ax2 = ax1.twinx()
    l4, = ax2.plot(plot_times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
    l7, = ax2.plot(plot_times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)
    ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)
    ax2.set_ylim(0, 1.0)

    # 温度曲线
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 65))
    l5, = ax3.plot(plot_times, plot_temperature, label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)
    ax3.set_ylabel('Environment Temperature/°C', color=colors[4], fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)
    ax3.set_ylim(-25, 40)

    # 阶段背景
    phase_split = args.max_time / 4
    ax1.axvspan(0, phase_split, alpha=0.2, color='lightblue', label='Flight Phase')
    ax1.axvspan(phase_split, 2*phase_split, alpha=0.2, color='lightgreen', label='Surface Sliding')
    ax1.axvspan(2*phase_split, 3*phase_split, alpha=0.2, color='salmon', label='Underwater Navigation')
    ax1.axvspan(3*phase_split, args.max_time, alpha=0.2, color='mediumpurple', label='Re-water Exit')

    # 图例配置
    lines = [l1, l2, l3, l6, l4, l7, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=LABEL_FONT_SIZE-2)
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    # 保存总图（SVG/PNG）
    save_path_svg = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Result.svg")
    save_path_png = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Result.png")
    plt.savefig(save_path_svg, bbox_inches='tight', dpi=1200)
    plt.savefig(save_path_png, dpi=1200, bbox_inches='tight')
    
    print(f"\n📊 原始测试结果图已保存:")
    print(f"   SVG: {save_path_svg}")
    print(f"   PNG: {save_path_png}")

    # --- 拆分图绘制 (multi_figures 子目录) ---
    multi_fig_dir = os.path.join(SAVE_DIR, "multi_figures")
    os.makedirs(multi_fig_dir, exist_ok=True)
    
    # 绘图通用配置
    fig_size = (15, 6)
    dpi_val = 1200
    grid_style = {'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.5}
    
    # 1. 第一幅图：功率需求和燃料电池输出功率 + 温度
    fig1, ax1_1 = plt.subplots(figsize=fig_size)
    fig1.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.95)
    
    ax1_1.plot(plot_times, plot_loads, label='Power Demand', color='#3570a8', alpha=LINES_ALPHA, linewidth=1.5)
    ax1_1.plot(plot_times, power_fc, label='Fuel Cell Power', color='#f09639', alpha=LINES_ALPHA, linewidth=1.5)

    # 温度曲线（右轴）
    ax1_2 = ax1_1.twinx()
    ax1_2.plot(plot_times, plot_temperature, label='Temperature', color='#8a7ab5', alpha=LINES_ALPHA, linewidth=1.5)
    ax1_2.set_ylabel('Temperature/°C', fontsize=LABEL_FONT_SIZE)
    ax1_2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE-2)
    ax1_2.set_ylim(-25, 40)
    
    # 配置坐标轴
    ax1_1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1_1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1_1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE-2)
    ax1_1.set_xlim(0, args.max_time)
    ax1_1.set_ylim(-2500, 5500)
    ax1_1.grid(**grid_style)
    
    # 图例
    lines1, labels1 = ax1_1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LABEL_FONT_SIZE-2, framealpha=0.9)
    
    # 保存图片
    fig1.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_FC_Power.svg"), 
                bbox_inches='tight', dpi=dpi_val)
    fig1.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_FC_Power.png"), 
                dpi=dpi_val, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. 第二幅图：锂电池输出功率和锂电池SOC
    fig2, ax2_1 = plt.subplots(figsize=fig_size)
    fig2.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.95)
    
    ax2_1.plot(plot_times, battery_power, label='Battery Power', color='#42985e', alpha=LINES_ALPHA, linewidth=1.5)
    ax2_1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax2_1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax2_1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE-2)
    ax2_1.set_xlim(0, args.max_time)
    ax2_1.set_ylim(-2500, 5500)
    ax2_1.grid(**grid_style)
    
    # SOC轴（右）
    ax2_2 = ax2_1.twinx()
    ax2_2.plot(plot_times, soc_bat, label='Battery SOC', color='#c84343', alpha=LINES_ALPHA, linewidth=1.5)
    ax2_2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax2_2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE-2)
    ax2_2.set_ylim(0, 1.0)
    
    # 合并图例
    lines1, labels1 = ax2_1.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LABEL_FONT_SIZE-2, framealpha=0.9)
    
    # 保存图片
    fig2.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_BAT_Power_SOC.svg"), 
                bbox_inches='tight', dpi=dpi_val)
    fig2.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_BAT_Power_SOC.png"), 
                dpi=dpi_val, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. 第三幅图：超级电容输出功率和超级电容SOC
    fig3, ax3_1 = plt.subplots(figsize=fig_size)
    fig3.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.95)
    
    ax3_1.plot(plot_times, power_sc, label='SuperCap Power', color='black', linestyle='--', alpha=LINES_ALPHA, linewidth=1.5)
    ax3_1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax3_1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax3_1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE-2)
    ax3_1.set_xlim(0, args.max_time)
    ax3_1.set_ylim(-2500, 5500)
    ax3_1.grid(**grid_style)
    
    # SOC轴（右）
    ax3_2 = ax3_1.twinx()
    ax3_2.plot(plot_times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA, linewidth=1.5)
    ax3_2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax3_2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE-2)
    ax3_2.set_ylim(0, 1.0)
    
    # 合并图例
    lines1, labels1 = ax3_1.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LABEL_FONT_SIZE-2, framealpha=0.9)
    
    # 保存图片
    fig3.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_SC_Power_SOC.svg"), 
                bbox_inches='tight', dpi=dpi_val)
    fig3.savefig(os.path.join(multi_fig_dir, f"{args.model_prefix}_SC_Power_SOC.png"), 
                dpi=dpi_val, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\n📊 拆分的三幅图已保存到 {multi_fig_dir}:")
    print(f"   1. FC_Power.svg/png (功率需求+燃料电池功率)")
    print(f"   2. BAT_Power_SOC.svg/png (锂电池功率+锂电池SOC)")
    print(f"   3. SC_Power_SOC.svg/png (超级电容功率+超级电容SOC)")

    # ====================== 保存JSON格式测试结果 ======================
    json_save_path = os.path.join(SAVE_DIR, f"{args.model_prefix}_Test_Results.json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    
    print(f"\n📄 JSON格式测试结果已保存:")
    print(f"   JSON: {json_save_path}")

    # ====================== 打印详细结果汇总 ======================
    print("\n" + "="*80)
    print("📈 JointNet测试结果汇总与分析")
    print("="*80)
    print(f"【等效氢耗】")
    print(f"  系统总等效氢耗：{total_h2:.6f} g")
    print(f"  ├─ 燃料电池氢耗：{total_fc_H2_g:.6f} g（{fc_h2_ratio*100:.2f}%）")
    print(f"  └─ 电池等效氢耗：{total_bat_H2_g:.6f} g（{bat_h2_ratio*100:.2f}%）")
    print(f"\n【电池 SOC 情况】")
    print(f"  电池 SOC 范围：{soc_bat_min:.6f} ~ {soc_bat_max:.6f}")
    print(f"  电池 SOC 变化幅度：{soc_bat_range:.6f}")
    print(f"\n【电池充电特性】")
    print(f"  充电步数：{bat_charge_steps} 步（{bat_charge_steps*dt:.2f}s）")
    print(f"  充电占比：{bat_charge_ratio*100:.2f}%")
    print(f"\n【超级电容特性】")
    print(f"  释放能量：{sc_release_Wh:.6f} Wh")
    print(f"  吸收能量：{sc_absorb_Wh:.6f} Wh")
    print(f"  未参与比例：{sc_inactive_ratio*100:.2f}%")
    print(f"\n【功率匹配性能】")
    print(f"  总未匹配功率（绝对值累加）：{total_unmatched_power:.6f} W·步")
    print(f"  平均未匹配功率：{avg_unmatched_power:.6f} W/步")
    print(f"  最大单次未匹配功率：{max_unmatched_power:.6f} W")
    print(f"  总未匹配能量：{total_unmatched_energy:.6f} Wh")
    print(f"  未匹配功率占总负载比例：{unmatched_ratio:.2f}%")
    print(f"\n【性能指标】")
    print(f"  累积奖励：{ep_r:.2f}")
    print(f"  总测试步数：{total_steps} 步")
    print(f"  总耗时：{total_time_cost:.4f}s")
    print(f"  平均步耗时：{avg_step_time:.6f}s/步")
    print("="*80)

    # 显示图像（根据命令行参数控制）
    if args.show_plot:
        plt.show()
    else:
        plt.close()  # 关闭图像释放内存
    
    print(f"\n✅ JointNet测试完成！所有结果已保存至：{SAVE_DIR}")
    print(f"   📄 JSON结果文件：{json_save_path}")
    print(f"   📊 原始可视化文件：{save_path_svg} / {save_path_png}")
    print(f"   📊 拆分图表文件：{multi_fig_dir}")