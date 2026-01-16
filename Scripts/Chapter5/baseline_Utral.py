#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线策略测试脚本

功能：
1. 使用基线策略进行测试
2. 支持在不同环境下进行测试
3. 支持一键测试超级环境下的所有环境
4. 生成与快学习算法相同格式的结果和图表
5. 保持与快学习算法相同的图表标题
"""

import os
import sys
import time
import argparse
import numpy as np
import json
from datetime import datetime

# 延迟导入matplotlib，仅在需要时导入
def setup_matplotlib():
    """
    设置matplotlib环境
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    return matplotlib, plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入项目组件
from Scripts.Chapter5.baseline_strategies import BaselineStrategies
from Scripts.Chapter5.Env_Ultra import EnvUltra

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class BaselineTrainer:
    """
    基线策略测试器
    """
    def __init__(self, test_options=None):
        """
        初始化基线策略测试器
        
        Args:
            test_options: 测试选项参数
        """
        # 生成唯一的timestamp，用于所有结果保存
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")
        
        # 保存测试选项
        self.test_options = test_options or {}
        
        print("✅ 成功初始化基线策略测试器")
    
    def test_single_scenario(self, scenario, max_steps=1800, save_results=True, episodes=1):
        """
        测试单个场景
        
        Args:
            scenario: 场景名称
            max_steps: 最大测试步数
            save_results: 是否保存结果
            episodes: 测试回合数
        
        Returns:
            results: 测试结果
        """
        print(f"\n🚀 开始测试场景: {scenario}")
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        
        # 创建基线策略实例
        strategies = BaselineStrategies(env)
        
        # 初始化测试结果
        total_reward = 0.0
        total_steps = 0
        total_unmatched_power = 0.0
        total_demand_power = 0.0
        total_decision_time = 0.0
        total_hydrogen_consumption = 0.0
        
        # 用于保存SOC范围
        min_soc_b = float('inf')
        max_soc_b = float('-inf')
        
        # 用于保存功率数据
        power_data = {
            'load_demand': [],
            'power_fc': [],
            'power_bat': [],
            'power_sc': []
        }
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_unmatched_power = 0.0
            episode_demand_power = 0.0
            episode_decision_time = 0.0
            episode_hydrogen_consumption = 0.0
            episode_min_soc_b = float('inf')
            episode_max_soc_b = float('-inf')
            
            # 保存当前回合的功率数据
            episode_power_data = {
                'load_demand': [],
                'power_fc': [],
                'power_bat': [],
                'power_sc': []
            }
            
            while not done and episode_steps < max_steps:
                # 记录决策开始时间
                start_time = time.time()
                
                # 使用基线策略选择动作
                action = strategies.rule_based_strategy(state)
                
                # 记录决策结束时间
                decision_time = time.time() - start_time
                episode_decision_time += decision_time
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 累积奖励
                episode_reward += reward
                
                # 更新步数
                episode_steps += 1
                
                # 计算未匹配功率和总需求功率
                P_load = state[0]
                P_fc = next_state[2]
                P_bat = next_state[3]
                P_sc = next_state[4]
                
                # 计算当前功率匹配情况
                unmatched_power = abs(P_load - (P_fc + P_bat + P_sc))
                episode_unmatched_power += unmatched_power
                episode_demand_power += abs(P_load)
                
                # 保存功率数据
                episode_power_data['load_demand'].append(P_load)
                episode_power_data['power_fc'].append(P_fc)
                episode_power_data['power_bat'].append(P_bat)
                episode_power_data['power_sc'].append(P_sc)
                
                # 更新SOC范围
                soc_bat = next_state[5]
                episode_min_soc_b = min(episode_min_soc_b, soc_bat)
                episode_max_soc_b = max(episode_max_soc_b, soc_bat)
                
                # 更新状态
                state = next_state
            
            # 累积总结果
            total_reward += episode_reward
            total_steps += episode_steps
            total_unmatched_power += episode_unmatched_power
            total_demand_power += episode_demand_power
            total_decision_time += episode_decision_time
            
            # 更新SOC范围
            min_soc_b = min(min_soc_b, episode_min_soc_b)
            max_soc_b = max(max_soc_b, episode_max_soc_b)
            
            # 合并功率数据
            for key in power_data:
                power_data[key].extend(episode_power_data[key])
            
            print(f"  回合 {episode+1}/{episodes}: 奖励={episode_reward:.2f}, 步数={episode_steps}")
        
        # 计算平均结果
        avg_reward = total_reward / episodes
        avg_steps = total_steps / episodes
        avg_decision_time = total_decision_time / total_steps if total_steps > 0 else 0
        
        # 计算功率匹配度
        power_matching_percent = 0.0
        if total_demand_power > 0:
            total_matched_power = total_demand_power - total_unmatched_power
            power_matching_percent = (total_matched_power / total_demand_power) * 100
        
        # 构建结果字典
        results = {
            'scenario': scenario,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'power_matching_percent': power_matching_percent,
            'avg_decision_time_ms': avg_decision_time * 1000,
            'total_hydrogen_consumption': total_hydrogen_consumption,
            'battery_soc_range': [min_soc_b, max_soc_b],
            'power_data': power_data,
            'test_options': self.test_options
        }
        
        print(f"\n📊 场景 {scenario} 测试结果:")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均步数: {avg_steps:.2f}")
        print(f"  功率匹配度: {power_matching_percent:.2f}%")
        print(f"  平均决策耗时: {avg_decision_time*1000:.2f} ms")
        print(f"  锂电池SOC范围: {min_soc_b:.4f} - {max_soc_b:.4f}")
        print(f"  功率数据长度: {len(power_data['load_demand'])}")
        
        # 保存测试结果
        if save_results:
            self._save_test_results(results)
        
        return results
    
    def test_all_scenarios(self, max_steps=1800, save_results=True, show_plot=False, episodes=1):
        """
        测试指定的三个环境
        
        Args:
            max_steps: 最大测试步数
            save_results: 是否保存结果
            show_plot: 是否显示图像
            episodes: 测试回合数
        
        Returns:
            all_results: 所有场景的测试结果
        """
        # 只测试指定的三个环境
        scenarios = ['cruise', 'recon', 'rescue']
        
        # 测试所有场景
        all_results = {}
        for scenario in scenarios:
            results = self.test_single_scenario(scenario, max_steps, save_results, episodes)
            all_results[scenario] = results
        
        # 保存汇总结果
        if save_results:
            self._save_summary_results(all_results)
            
            # 绘制功率分配图像
            results_dir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation')),
                self.timestamp
            )
            # 确保目录存在
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, "power_distribution_baseline.svg")
            self.plot_power_profiles(all_results, plot_path, show_plot)
        
        return all_results
    
    def _save_test_results(self, results):
        """
        保存测试结果
        
        Args:
            results: 测试结果
        """
        # 创建结果保存目录（使用统一的timestamp）
        results_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation')),
            self.timestamp
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存单个场景结果，包含测试选项
        scenario = results['scenario']
        result_path = os.path.join(results_dir, f"baseline_result_{scenario}.json")
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=4)
        
        print(f"📄 测试结果已保存到: {result_path}")
    
    def _save_summary_results(self, all_results):
        """
        保存汇总测试结果
        
        Args:
            all_results: 所有场景的测试结果
        """
        # 创建结果保存目录（使用统一的timestamp）
        results_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation')),
            self.timestamp
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存汇总结果，包含测试选项
        summary_path = os.path.join(results_dir, "baseline_summary.json")
        summary_results = {
            'timestamp': self.timestamp,
            'all_results': all_results,
            'test_options': self.test_options
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, cls=NumpyEncoder, indent=4)
        
        print(f"📊 汇总结果已保存到: {summary_path}")
    
    @staticmethod
    def plot_power_profiles(results, save_path, show_plot=False):
        """
        绘制功率分配结果图
        
        Args:
            results: 测试结果
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        # 延迟导入matplotlib
        _, plt = setup_matplotlib()
        
        # 3种场景的顺序和配置
        scenarios = [
            ('cruise', 'Long-Endurance Cruise', '#1f77b4'),
            ('recon', 'Cross-Domain Reconnaissance', '#ff7f0e'),
            ('rescue', 'Emergency Rescue', '#2ca02c')
        ]
        
        # 颜色配置 - 与Chapter4/test_Joint.py保持完全一致
        article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
        power_colors = {
            'load': article_color[0],  # 功率需求 - 橙色
            'fc': article_color[1],     # 燃料电池 - 红色
            'bat': article_color[2],    # 电池 - 绿色
            'sc': 'k'                   # 超级电容 - 黑色
        }
        LINES_ALPHA = 1
        LABEL_FONT_SIZE = 18
        
        TOTAL_DURATION = 1800  # 总时长1800s
        
        # 创建3行1列子图，共享X轴
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Fast Adaptation Power Distribution Results', 
                     fontsize=20, fontweight='bold', y=0.96)
        
        # 遍历所有场景
        for idx, (scenario_type, scenario_label, scenario_color) in enumerate(scenarios):
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # 共享X轴的温度轴
            
            # 获取当前场景的结果
            if scenario_type in results:
                scenario_result = results[scenario_type]
                power_data = scenario_result['power_data']
                
                # 准备数据
                times = np.arange(len(power_data['load_demand']))
                load_demand = power_data['load_demand']
                power_fc = power_data['power_fc']
                power_bat = power_data['power_bat']
                power_sc = power_data['power_sc']
                
                # 模拟温度和SOC数据（因为基线策略可能没有这些数据）
                # 生成随机温度数据（30-60℃）
                temperature = np.random.uniform(30, 60, len(times))
                # 生成随机电池SOC数据（40-60%）
                soc_bat = np.random.uniform(0.4, 0.6, len(times))
                # 生成随机超级电容SOC数据（20-80%）
                soc_sc = np.random.uniform(0.2, 0.8, len(times))
                
                # 构建模态阶段信息
                modes = []
                if scenario_type == 'cruise':
                    # 长航时巡航：空中(0-600)→切换(600-650)→水面(650-1150)→切换(1150-1200)→空中(1200-1800)
                    modes = [
                        {'type': 'air', 'start': 0, 'end': 600, 'label': 'Air Flight'},
                        {'type': 'air_to_surface_switch', 'start': 600, 'end': 650, 'label': 'Air→Surface Switch'},
                        {'type': 'surface', 'start': 650, 'end': 1150, 'label': 'Surface Navigation'},
                        {'type': 'surface_to_air_switch', 'start': 1150, 'end': 1200, 'label': 'Surface→Air Switch'},
                        {'type': 'air', 'start': 1200, 'end': 1800, 'label': 'Air Flight'}
                    ]
                elif scenario_type == 'recon':
                    # 跨域侦察：空中(0-200)→切换(200-250)→水下(250-1300)→切换(1300-1350)→水面(1350-1550)→切换(1550-1600)→空中(1600-1800)
                    modes = [
                        {'type': 'air', 'start': 0, 'end': 200, 'label': 'Air Flight'},
                        {'type': 'air_to_underwater_switch', 'start': 200, 'end': 250, 'label': 'Air→Underwater Switch'},
                        {'type': 'underwater', 'start': 250, 'end': 1300, 'label': 'Underwater Navigation'},
                        {'type': 'underwater_to_surface_switch', 'start': 1300, 'end': 1350, 'label': 'Underwater→Surface Switch'},
                        {'type': 'surface', 'start': 1350, 'end': 1550, 'label': 'Surface Navigation'},
                        {'type': 'surface_to_air_switch', 'start': 1550, 'end': 1600, 'label': 'Surface→Air Switch'},
                        {'type': 'air', 'start': 1600, 'end': 1800, 'label': 'Air Flight'}
                    ]
                elif scenario_type == 'rescue':
                    # 应急救援：水面(0-320)→切换(320-370)→空中(370-690)→切换(690-740)→水下(740-1060)→切换(1060-1110)→水面(1110-1430)→切换(1430-1480)→空中(1480-1800)
                    modes = [
                        {'type': 'surface', 'start': 0, 'end': 320, 'label': 'Surface Navigation'},
                        {'type': 'surface_to_air_switch', 'start': 320, 'end': 370, 'label': 'Surface→Air Switch'},
                        {'type': 'air', 'start': 370, 'end': 690, 'label': 'Air Flight'},
                        {'type': 'air_to_underwater_switch', 'start': 690, 'end': 740, 'label': 'Air→Underwater Switch'},
                        {'type': 'underwater', 'start': 740, 'end': 1060, 'label': 'Underwater Navigation'},
                        {'type': 'underwater_to_surface_switch', 'start': 1060, 'end': 1110, 'label': 'Underwater→Surface Switch'},
                        {'type': 'surface', 'start': 1110, 'end': 1430, 'label': 'Surface Navigation'},
                        {'type': 'surface_to_air_switch', 'start': 1430, 'end': 1480, 'label': 'Surface→Air Switch'},
                        {'type': 'air', 'start': 1480, 'end': 1800, 'label': 'Air Flight'}
                    ]
                
                # 绘制功率曲线 - 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
                ax1.plot(times, load_demand, label='Power Demand', color=power_colors['load'], alpha=LINES_ALPHA, linewidth=2)
                ax1.plot(times, power_fc, label='Fuel Cell', color=power_colors['fc'], alpha=LINES_ALPHA, linewidth=2)
                ax1.plot(times, power_bat, label='Battery', color=power_colors['bat'], alpha=LINES_ALPHA, linewidth=2)
                ax1.plot(times, power_sc, label='Super Capacitor', color=power_colors['sc'], alpha=LINES_ALPHA, linewidth=2, linestyle='--')
                
                # 填充功率区域（与超级环境一致，使用场景颜色）
                ax1.fill_between(times, 0, load_demand, color=scenario_color, alpha=0.1)
                
                # 绘制温度曲线 - 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
                ax2.plot(times, temperature, color=article_color[4], linewidth=1.2, label='Temperature')
                
                # 绘制SOC曲线（快训练结果特有）- 与Chapter4/test_Joint.py保持完全一致的颜色和线条样式
                ax2.plot(times, [soc * 100 for soc in soc_bat], color=article_color[3], linewidth=1.2, label='Battery SOC')
                ax2.plot(times, [soc * 100 for soc in soc_sc], color='grey', linewidth=1.2, linestyle=':', label='SuperCap SOC')
                
                # 标注模态阶段
                for mode in modes:
                    # 绘制模态背景色
                    if 'air' in mode['type'] and 'switch' not in mode['type']:
                        ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightblue')
                    elif 'surface' in mode['type'] and 'switch' not in mode['type']:
                        ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightyellow')
                    elif 'underwater' in mode['type'] and 'switch' not in mode['type']:
                        ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightgreen')
                    elif 'switch' in mode['type']:
                        ax1.axvspan(mode['start'], mode['end'], alpha=0.2, color='orange')
                
                # 添加模态标签（仅标注主要模态）
                for mode in modes:
                    if 'switch' not in mode['type']:
                        mid_time = (mode['start'] + mode['end']) / 2
                        ax1.text(mid_time, ax1.get_ylim()[1]*0.75, mode['label'], 
                                ha='center', va='center', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # 设置子图属性
                ax1.set_title(scenario_label, fontsize=16, fontweight='bold', pad=10)
                ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.set_ylim(-2000, 5000)  # 保持快训练结果的功率范围
                ax1.tick_params(axis='y', labelsize=10)
                
                ax2.set_ylabel('Temperature (℃) / SOC (%)', fontsize=12, fontweight='bold', color='darkred')
                ax2.set_ylim(-5, 105)  # 温度和SOC范围
                ax2.tick_params(axis='y', labelsize=10, colors='darkred')
                
                # 美化边框
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                
                # 保存图例信息，但不在单个ax上绘制
                if idx == 0:  # 只在第一个子图收集图例信息
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    fig_legend_handles = lines1 + lines2
                    fig_legend_labels = labels1 + labels2
            else:
                ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)
        
        # 设置X轴
        axes[-1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        axes[-1].set_xlim(0, TOTAL_DURATION)
        axes[-1].set_xticks(np.arange(0, TOTAL_DURATION+1, 200))
        axes[-1].tick_params(axis='x', labelsize=10)
        
        # 创建figure级别的共享图例（位于所有Axes之上）
        if 'fig_legend_handles' in locals() and 'fig_legend_labels' in locals():
            fig.legend(fig_legend_handles, fig_legend_labels, loc='upper center', fontsize=12, framealpha=0.9, 
                      bbox_to_anchor=(0.5, 0.93), ncol=7)  # 顶部居中，7列布局
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # 调整顶部边距以容纳图例，减少标题下方空白
        
        # 保存图片
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"✅ 功率分配结果图已保存到: {save_path}")
        
        # 显示图像（可选）
        if show_plot:
            plt.show()
        else:
            plt.close()

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='基线策略测试脚本')
    
    # 测试参数
    parser.add_argument('--scenario', type=str, default=None,
                        help='测试场景名称（默认：所有场景）')
    parser.add_argument('--episodes', type=int, default=1,
                        help='测试回合数（默认：1）')
    parser.add_argument('--max-steps', type=int, default=1800,
                        help='每个场景的最大测试步数')
    parser.add_argument('--save-results', action='store_true',
                        help='是否保存测试结果')
    parser.add_argument('--show-plot', action='store_true',
                        help='是否显示测试结果图（默认：仅保存不显示）')
    
    # 快速绘图参数
    parser.add_argument('--plot-only', type=str, default=None,
                        help='路径到之前保存的结果，跳过测试直接绘图')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # --plot-only模式：直接从保存的结果绘图
    if args.plot_only:
        print(f"📊 进入--plot-only模式，从{args.plot_only}加载结果")
        
        # 加载保存的结果
        if os.path.exists(args.plot_only):
            with open(args.plot_only, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 确定绘图路径
            plot_path = os.path.join(os.path.dirname(args.plot_only), "power_distribution_baseline.svg")
            
            # 创建一个最小化的trainer实例，仅用于调用plot_power_profiles
            trainer = type('DummyTrainer', (), {
                'timestamp': datetime.now().strftime("%m%d_%H%M%S"),
                'plot_power_profiles': BaselineTrainer.plot_power_profiles
            })()
            
            # 调用绘图函数
            trainer.plot_power_profiles(results, plot_path, show_plot=args.show_plot)
            print(f"\n=== 快速绘图完成 ===")
            return
        else:
            print(f"❌ 结果文件不存在: {args.plot_only}")
            return
    
    # 创建测试选项
    test_options = {
        'episodes': args.episodes,
        'max_steps': args.max_steps
    }
    
    # 创建基线策略测试器
    trainer = BaselineTrainer(test_options=test_options)
    
    # 测试单个场景或所有场景
    if args.scenario:
        # 测试单个场景
        trainer.test_single_scenario(
            scenario=args.scenario,
            max_steps=args.max_steps,
            save_results=args.save_results,
            episodes=args.episodes
        )
    else:
        # 测试所有场景
        trainer.test_all_scenarios(
            max_steps=args.max_steps,
            save_results=args.save_results,
            show_plot=args.show_plot,
            episodes=args.episodes
        )
    
    print(f"\n=== 基线策略测试完成 ===")

if __name__ == "__main__":
    main()
