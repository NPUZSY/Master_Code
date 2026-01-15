import numpy as np
import torch
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式
import matplotlib.pyplot as plt
import time

# 添加项目根目录到Python路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入环境
from Scripts.Chapter5.Env_Ultra import EnvUltra
from Scripts.utils.global_utils import font_get

# 设置字体
font_get()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------
# 基准策略类
# ----------------------------------------------------
class BaselineStrategies:
    """
    基准策略类，包含基于规则的策略和基于DP的策略
    """
    def __init__(self, env):
        self.env = env
        
    def rule_based_strategy(self, state):
        """
        基于规则的策略：
        基线策略以锂电池SOC和负载功率需求为核心输入，通过层级化规则实现燃料电池、锂电池及超级电容的功率动态调整
        
        Args:
            state: 环境状态
        
        Returns:
            action_list: 动作列表 [fc_action, bat_action, sc_action]
        """
        P_load = state[0]
        current_fc_power = state[2]
        soc_bat = state[5]  # 当前锂电池SOC
        
        # FC动作处理逻辑：
        # 1. 强化学习算法输出的是0-31的索引值
        # 2. 环境将索引转换为实际动作值k：k = K_FC_MIN + idx，其中K_FC_MIN=-15
        # 3. 然后转换为功率变化：delta = k * 0.01 * P_fc_max
        
        P_fc_max = self.env.P_FC_MAX
        K_FC_MIN = self.env.K_FC_MIN  # -15
        N_FC_ACTIONS = self.env.N_FC_ACTIONS  # 32
        
        # 生成所有可能的动作索引和对应的功率输出
        possible_actions = []
        for idx in range(N_FC_ACTIONS):
            # 将索引转换为实际动作值
            k = K_FC_MIN + idx
            # 计算功率变化
            delta_P = k * 0.01 * P_fc_max
            # 计算新的功率输出
            new_power = current_fc_power + delta_P
            new_power = np.clip(new_power, 0, P_fc_max)
            # 保存(索引, 功率输出)二元组
            possible_actions.append((idx, new_power))
        
        # 根据SOC状态选择最佳动作索引
        best_idx = 0
        best_power = current_fc_power
        
        # 1. 亏电状态：SOC < 0.2
        if soc_bat < 0.2:
            # 燃料电池全速提升输出功率
            # 选择最大的功率输出对应的索引
            best_idx, best_power = max(possible_actions, key=lambda x: x[1])
        
        # 2. 低电量状态：0.2 ≤ SOC < 0.5
        elif 0.2 <= soc_bat < 0.5:
            # 燃料电池使用大于功率需求的最小档位输出功率
            # 选择大于等于P_load的最小功率输出对应的索引
            candidates = [item for item in possible_actions if item[1] >= P_load]
            if candidates:
                # 找到大于等于P_load的最小功率
                best_idx, best_power = min(candidates, key=lambda x: x[1])
            else:
                # 如果没有大于等于P_load的功率，选择最大的功率
                best_idx, best_power = max(possible_actions, key=lambda x: x[1])
        
        # 3. 理想SOC范围：0.5 ≤ SOC < 0.7
        elif 0.5 <= soc_bat < 0.7:
            # 燃料电池使用最靠近功率需求的档位输出
            # 找到最接近P_load的功率对应的索引
            best_idx, best_power = min(possible_actions, key=lambda x: abs(x[1] - P_load))
        
        # 4. 高电量状态：0.7 ≤ SOC < 0.9
        elif 0.7 <= soc_bat < 0.9:
            # 燃料电池使用小于功率需求的最大档位输出功率
            # 选择小于等于P_load的最大功率输出对应的索引
            candidates = [item for item in possible_actions if item[1] <= P_load]
            if candidates:
                # 找到小于等于P_load的最大功率
                best_idx, best_power = max(candidates, key=lambda x: x[1])
            else:
                # 如果没有小于等于P_load的功率，选择最小的功率
                best_idx, best_power = min(possible_actions, key=lambda x: x[1])
        
        # 5. 满电状态：SOC ≥ 0.9
        else:  # soc_bat >= 0.9
            # 燃料电池全速降低输出功率
            # 选择最小的功率输出对应的索引
            best_idx, best_power = min(possible_actions, key=lambda x: x[1])
        
        # 确保best_idx是整数
        best_idx = int(best_idx)
        
        # 计算剩余功率需求，由锂电池补充
        remaining_power = P_load - best_power
        
        # 锂电池动作处理逻辑：
        # 1. 强化学习算法输出的是0-39的索引值
        # 2. 环境将索引转换为实际动作值k：k = K_BAT_MIN + idx，其中K_BAT_MIN=-20
        # 3. 然后转换为功率：p = k * 0.05 * P_BAT_MAX
        # 4. 我们需要将剩余功率转换为索引值
        
        P_bat_max = self.env.P_BAT_MAX
        K_BAT_MIN = self.env.K_BAT_MIN  # -20
        K_BAT_MAX = self.env.K_BAT_MAX  # 19
        N_BAT_ACTIONS = self.env.N_BAT_ACTIONS  # 40
        
        # 计算所需的动作值k
        desired_k = remaining_power / (0.05 * P_bat_max)
        
        # 将动作值转换为索引值
        # 索引 = 动作值 - K_BAT_MIN
        bat_idx = int(np.round(desired_k)) - K_BAT_MIN
        
        # 确保索引在合法范围内
        bat_idx = np.clip(bat_idx, 0, N_BAT_ACTIONS - 1)
        bat_idx = int(bat_idx)
        
        # 计算锂电池能提供的功率
        bat_power = self.env._bat_power_from_index(bat_idx)
        bat_power = np.clip(bat_power, -self.env.P_BAT_MAX, self.env.P_BAT_MAX)
        
        # 计算最终的功率差
        final_power_diff = P_load - best_power - bat_power
        
        # 超级电容仅在锂电池SOC过高或过低时接入系统
        # 亏电状态（SOC < 0.2）或满电状态（SOC >= 0.9）        
        sc_action = 1 
        
        return [best_idx, bat_idx, sc_action]
    


# ----------------------------------------------------
# 测试脚本
# ----------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基准策略测试脚本')
    parser.add_argument('--scenario', type=str, default='cruise', 
                        choices=EnvUltra.SCENARIO_TYPES, 
                        help='测试场景类型')
    parser.add_argument('--strategy', type=str, default='rule_based', 
                        choices=['rule_based'], 
                        help='使用的策略类型')
    parser.add_argument('--output-dir', type=str, default='', 
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not args.output_dir:
        timestamp = time.strftime("%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "nets", "Chap5", "baseline_results", timestamp)
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化环境
    env = EnvUltra(scenario_type=args.scenario)
    
    # 初始化策略
    strategies = BaselineStrategies(env)
    
    # 测试策略
    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    # 保存测试数据
    power_data = {
        'power_fc': [],
        'power_bat': [],
        'power_sc': [],
        'load_power': [],
        'soc_bat': [],
        'soc_sc': [],
        'temperature': [],
        'rewards': []
    }
    
    info_data = []
    
    while not done:
        # 选择动作
        if args.strategy == 'rule_based':
            action_list = strategies.rule_based_strategy(state)
        else:
            action_list = strategies.dp_strategy(state)
        
        # 执行动作
        next_state, reward, done, info = env.step(action_list)
        
        # 保存数据
        total_reward += reward
        step_count += 1
        
        # 保存功率数据
        power_data['power_fc'].append(info['P_fc'])
        power_data['power_bat'].append(info['P_bat'])
        power_data['power_sc'].append(info['P_sc'])
        power_data['load_power'].append(info['P_load'])
        power_data['soc_bat'].append(state[5])
        power_data['soc_sc'].append(info['soc_sc'])
        power_data['temperature'].append(info['T_amb'])
        power_data['rewards'].append(reward)
        
        # 保存完整信息
        info_data.append(info)
        
        # 更新状态
        state = next_state
    
    # 保存结果
    print(f"\n=== 测试结果 ===")
    print(f"场景类型: {args.scenario}")
    print(f"策略类型: {args.strategy}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"总步数: {step_count}")
    print(f"平均每步奖励: {total_reward / step_count:.4f}")
    
    # 1. 保存测试结果到JSON文件
    test_results = {
        'scenario_type': args.scenario,
        'strategy_type': args.strategy,
        'total_reward': float(total_reward),
        'total_steps': step_count,
        'average_reward_per_step': float(total_reward / step_count),
        'power_data': power_data,
        'info_data': info_data
    }
    
    json_path = os.path.join(output_dir, f"test_result_{args.scenario}_{args.strategy}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # 自定义JSON编码器，处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        json.dump(test_results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✅ 测试结果保存到: {json_path}")
    
    # 2. 生成并保存功率分配图
    # 生成时间轴
    times = np.arange(len(power_data['power_fc']))
    
    # 绘图配置 - 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式配置
    article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    power_colors = {
        'load': article_color[0],  # 功率需求 - 橙色
        'fc': article_color[1],     # 燃料电池 - 红色
        'bat': article_color[2],    # 电池 - 绿色
        'sc': 'k'                   # 超级电容 - 黑色
    }
    colors = article_color
    LINES_ALPHA = 1
    LABEL_FONT_SIZE = 18
    
    # --- 总图绘制 --- 参考test_Joint.py的布局
    fig, ax1 = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875)
    
    # 功率曲线 - 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
    l1, = ax1.plot(times, power_data['load_power'], label='Power Demand', color=power_colors['load'], alpha=LINES_ALPHA, linewidth=2)
    l2, = ax1.plot(times, power_data['power_fc'], label='Power Fuel Cell', color=power_colors['fc'], alpha=LINES_ALPHA, linewidth=2)
    l3, = ax1.plot(times, power_data['power_bat'], label='Power Battery', color=power_colors['bat'], alpha=LINES_ALPHA, linewidth=2)
    l6, = ax1.plot(times, power_data['power_sc'], label='Power SuperCap', color=power_colors['sc'], alpha=LINES_ALPHA, linewidth=2, linestyle='--')
    
    # 配置主坐标轴（功率轴）
    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax1.set_xlim(0, len(times))
    ax1.set_ylim(-2500, 5500)
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_title(f'Power Distribution - {args.scenario} Scenario - {args.strategy} Strategy', fontsize=16, fontweight='bold')
    
    # SOC曲线（右轴1）- 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
    ax2 = ax1.twinx()
    l4, = ax2.plot(times, power_data['soc_bat'], label='Battery SOC', color=article_color[3], alpha=LINES_ALPHA, linewidth=1.5)
    l7, = ax2.plot(times, power_data['soc_sc'], label='SuperCap SOC', color='grey', alpha=LINES_ALPHA, linewidth=1.5, linestyle=':')
    ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)
    ax2.set_ylim(0, 1.0)
    
    # 温度曲线（右轴2，向外偏移）- 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 65))
    l5, = ax3.plot(times, power_data['temperature'], label='Environment Temperature', color=article_color[4], alpha=LINES_ALPHA, linewidth=1.5)
    ax3.set_ylabel('Environment Temperature/°C', color=article_color[4], fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='y', labelcolor=article_color[4], labelsize=LABEL_FONT_SIZE)
    ax3.set_ylim(-25, 40)
    
    # 绘制模态背景
    for mode in env.mode_annotations:
        start = mode['start']
        end = mode['end']
        mode_type = mode['type']
        
        # 模态到颜色的映射
        mode_colors = {
            'air': ('lightblue', 'Flight Phase'),
            'surface': ('lightgreen', 'Surface Sliding'),
            'underwater': ('salmon', 'Underwater Navigation'),
            'air_to_surface_switch': ('lightblue', 'Air to Surface'),
            'surface_to_air_switch': ('lightgreen', 'Surface to Air'),
            'air_to_underwater_switch': ('lightblue', 'Air to Underwater'),
            'underwater_to_surface_switch': ('salmon', 'Underwater to Surface'),
            'surface_to_underwater_switch': ('lightgreen', 'Surface to Underwater'),
            'underwater_to_air_switch': ('salmon', 'Underwater to Air')
        }
        
        if mode_type in mode_colors:
            color, label = mode_colors[mode_type]
            ax1.axvspan(start, end, alpha=0.2, color=color)
    
    # 图例配置 - 放在底部，参考test_Joint.py的设置
    lines = [l1, l2, l3, l6, l4, l7, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=LABEL_FONT_SIZE-2)
    
    # 保存图像
    svg_path = os.path.join(output_dir, f"power_distribution_{args.scenario}_{args.strategy}.svg")
    png_path = os.path.join(output_dir, f"power_distribution_{args.scenario}_{args.strategy}.png")
    
    plt.savefig(svg_path, bbox_inches='tight', dpi=1200)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 功率分配图保存到:")
    print(f"   SVG: {svg_path}")
    print(f"   PNG: {png_path}")
    
    # 关闭图像
    plt.close()
    
    print(f"\n🎉 所有结果已保存到: {output_dir}")
