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
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# 导入项目组件
from Scripts.Chapter5.Meta_RL_Engine import MetaRLPolicy
from Scripts.Chapter3.MARL_Engine import device
from Scripts.Chapter5.Env_Ultra import EnvUltra

# ====================== 2. 工具类与参数解析 ======================
class NumpyEncoder(JSONEncoder):
    """自定义JSON编码器，处理numpy类型和其他非标准类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='慢学习模型测试脚本')
    
    # 核心：模型路径参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='训练好的模型路径（必填）')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='隐藏层维度（默认：512）')
    
    # 可选配置参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')
    parser.add_argument('--max-steps', type=int, default=1800, help='每个模态的最大测试步数（默认：1800）')
    parser.add_argument('--episodes', type=int, default=1, help='测试回合数（默认：1）')
    parser.add_argument('--show-plot', action='store_true', help='是否显示测试结果图（默认：仅保存不显示）')
    parser.add_argument('--save-dir', type=str, default=None, help='结果保存目录（默认：模型所在目录）')
    
    return parser.parse_args()

# ====================== 3. 测试核心功能 ======================
def test_single_scenario(model, scenario, max_steps=1800, seed=42, episodes=1):
    """测试单个场景，支持多回合"""
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 初始化总统计
    total_reward = 0.0
    total_steps = 0
    
    # 保存所有回合的数据
    all_episodes_data = []
    
    for episode in range(episodes):
        print(f"\n--- 回合 {episode+1}/{episodes} ---")
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        state = env.reset()
        
        # 初始化数据收集列表
        times = []
        power_fc = []
        power_bat = []
        power_sc = []
        load_demand = []
        temperature = []
        soc_bat = []
        soc_sc = []
        rewards = []
        
        episode_reward = 0.0
        episode_steps = 0
        
        while episode_steps < max_steps:
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
            fc_action_out, bat_action_out, sc_action_out, _ = model(state_tensor, None)
            
            # 贪婪选择动作
            fc_action = torch.argmax(fc_action_out, dim=1).item()
            bat_action = torch.argmax(bat_action_out, dim=1).item()
            sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            # 记录数据
            times.append(episode_steps)
            power_fc.append(info['P_fc'])
            power_bat.append(info['P_bat'])
            power_sc.append(info['P_sc'])
            load_demand.append(info['P_load'])
            temperature.append(info['T_amb'])
            soc_bat.append(next_state[5])  # 假设state[5]是电池SOC
            soc_sc.append(next_state[6])  # 假设state[6]是超级电容SOC
            rewards.append(reward)
            
            episode_reward += reward
            state = next_state
            episode_steps += 1
            
            if done:
                break
        
        # 更新总统计
        total_reward += episode_reward
        total_steps += episode_steps
        
        # 保存回合数据
        all_episodes_data.append({
            "episode": episode+1,
            "times": times,
            "power_fc": power_fc,
            "power_bat": power_bat,
            "power_sc": power_sc,
            "load_demand": load_demand,
            "temperature": temperature,
            "soc_bat": soc_bat,
            "soc_sc": soc_sc,
            "rewards": rewards,
            "total_reward": episode_reward,
            "steps": episode_steps
        })
        
        print(f"✅ 回合 {episode+1} 完成，奖励: {episode_reward:.2f}，步数: {episode_steps}")
    
    # 计算统计指标
    avg_reward = total_reward / total_steps if total_steps > 0 else 0.0
    
    # 计算功率不匹配度（只使用第一个回合的数据，因为绘图需要）
    if all_episodes_data:
        first_episode = all_episodes_data[0]
        total_unmatched_power = sum(abs(ld - (fc + bat + sc)) for ld, fc, bat, sc in zip(
            first_episode['load_demand'], 
            first_episode['power_fc'], 
            first_episode['power_bat'], 
            first_episode['power_sc']
        ))
        avg_unmatched_power = total_unmatched_power / first_episode['steps'] if first_episode['steps'] > 0 else 0.0
    else:
        total_unmatched_power = 0.0
        avg_unmatched_power = 0.0
    
    test_results = {
        "scenario": scenario,
        "total_steps": total_steps,
        "total_reward": total_reward,
        "average_reward": avg_reward,
        "total_unmatched_power": total_unmatched_power,
        "average_unmatched_power": avg_unmatched_power,
        "episodes": episodes,
        "raw_data": {
            "times": first_episode['times'] if all_episodes_data else [],
            "power_fc": first_episode['power_fc'] if all_episodes_data else [],
            "power_bat": first_episode['power_bat'] if all_episodes_data else [],
            "power_sc": first_episode['power_sc'] if all_episodes_data else [],
            "load_demand": first_episode['load_demand'] if all_episodes_data else [],
            "temperature": first_episode['temperature'] if all_episodes_data else [],
            "soc_bat": first_episode['soc_bat'] if all_episodes_data else [],
            "soc_sc": first_episode['soc_sc'] if all_episodes_data else [],
            "rewards": first_episode['rewards'] if all_episodes_data else []
        },
        "all_episodes": all_episodes_data
    }
    
    return test_results

# ====================== 4. 可视化功能 ======================
def plot_power_profiles(results, save_path, show_plot=False):
    """绘制9种模态的功率分配结果，3x3子图"""
    # 9种场景的顺序
    scenarios = [
        'air', 'surface', 'underwater',  # 3种基础场景
        'air_to_surface', 'surface_to_air',  # 切换场景1-2
        'air_to_underwater', 'underwater_to_air',  # 切换场景3-4
        'surface_to_underwater', 'underwater_to_surface'  # 切换场景5-6
    ]
    
    # 创建3x3子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle('Power Distribution Results for 9 Scenarios', fontsize=20, fontweight='bold')
    
    # 颜色配置
    colors = {
        'load': '#f09639',  # 功率需求
        'fc': '#c84343',     # 燃料电池
        'bat': '#42985e',    # 电池
        'sc': '#8a7ab5'      # 超级电容
    }
    
    # 模态背景色映射（使用RGBA颜色值，与超级环境Fig5-7保持一致的透明度）
    background_colors = {
        'air': (0.878, 0.925, 0.973, 0.7),  # lightblue with alpha=0.1
        'surface': (1.0, 1.0, 0.902, 0.7),   # lightyellow with alpha=0.1
        'underwater': (0.941, 0.973, 0.859, 0.7),  # lightgreen with alpha=0.1
        'air_to_surface': (1.0, 0.647, 0.0, 0.2),  # orange with alpha=0.2
        'surface_to_air': (1.0, 0.647, 0.0, 0.2),  # orange with alpha=0.2
        'air_to_underwater': (1.0, 0.647, 0.0, 0.2),  # orange with alpha=0.2
        'underwater_to_air': (1.0, 0.647, 0.0, 0.2),  # orange with alpha=0.2
        'surface_to_underwater': (1.0, 0.647, 0.0, 0.2),  # orange with alpha=0.2
        'underwater_to_surface': (1.0, 0.647, 0.0, 0.2)   # orange with alpha=0.2
    }
    
    # 绘制每个子图
    for i, scenario in enumerate(scenarios):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 获取当前场景的结果
        scenario_result = results[scenario]
        data = scenario_result['raw_data']
        
        # 设置子图背景色
        ax.set_facecolor(background_colors.get(scenario, 'white'))
        
        # 绘制功率曲线 - 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
        ax.plot(data['times'], data['load_demand'], label='Load Demand', color=colors['load'], alpha=1, linewidth=2)
        ax.plot(data['times'], data['power_fc'], label='Fuel Cell', color=colors['fc'], alpha=1, linewidth=2)
        ax.plot(data['times'], data['power_bat'], label='Battery', color=colors['bat'], alpha=1, linewidth=2)
        ax.plot(data['times'], data['power_sc'], label='Super Capacitor', color=colors['sc'], alpha=1, linewidth=2, linestyle='--')
        
        # 配置子图
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置轴标签
        if row == 2:  # 最后一行
            ax.set_xlabel('Time (s)', fontsize=10)
        if col == 0:  # 第一列
            ax.set_ylabel('Power (W)', fontsize=10)
        
        # 设置轴范围
        ax.set_xlim(0, len(data['times'])-1)
        max_power = max(max(data['load_demand']), max(data['power_fc']), max(data['power_bat']), max(data['power_sc']))
        min_power = min(min(data['load_demand']), min(data['power_fc']), min(data['power_bat']), min(data['power_sc']))
        ax.set_ylim(min_power * 1.1, max_power * 1.1)
    
    # 添加全局图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 保存图片
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    print(f"✅ 功率分配结果图已保存到: {save_path}")
    
    # 显示图像（可选）
    if show_plot:
        plt.show()
    else:
        plt.close()

# ====================== 5. 主测试程序 ======================
def main():
    args = parse_args()
    
    # 打印配置确认信息
    print("=" * 80)
    print("                    慢学习模型测试配置确认                  ")
    print("=" * 80)
    print(f"待测试模型路径: {args.model_path}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"随机种子: {args.seed}")
    print(f"每个模态的最大测试步数: {args.max_steps}")
    print(f"显示结果图: {'是' if args.show_plot else '否'}")
    print("=" * 80 + "\n")
    
    # 设置保存目录
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.dirname(args.model_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    try:
        model = MetaRLPolicy(hidden_dim=args.hidden_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"✅ 成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 9种场景
    scenarios = [
        'air', 'surface', 'underwater',
        'air_to_surface', 'surface_to_air',
        'air_to_underwater', 'underwater_to_air',
        'surface_to_underwater', 'underwater_to_surface'
    ]
    
    # 测试所有场景
    test_results = {}
    for scenario in scenarios:
        print(f"🚀 测试场景: {scenario}")
        result = test_single_scenario(model, scenario, max_steps=args.max_steps, seed=args.seed, episodes=args.episodes)
        test_results[scenario] = result
        
        # 保存单个场景的JSON结果
        scenario_json_path = os.path.join(save_dir, f"test_result_{scenario}.json")
        with open(scenario_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
        print(f"✅ 场景 {scenario} 测试完成，结果已保存到: {scenario_json_path}")
    
    # 保存所有场景的汇总JSON结果
    all_results_path = os.path.join(save_dir, "test_results_all.json")
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    print(f"✅ 所有场景测试结果已保存到: {all_results_path}")
    
    # 绘制3x3功率分配结果图
    plot_path = os.path.join(save_dir, "power_distribution_9_scenarios.svg")
    plot_power_profiles(test_results, plot_path, show_plot=args.show_plot)
    
    print(f"\n✅ 慢学习模型测试完成！所有结果已保存到: {save_dir}")

# ====================== 6. 入口函数 ======================
if __name__ == '__main__':
    main()
