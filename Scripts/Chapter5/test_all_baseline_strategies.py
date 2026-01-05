#!/usr/bin/env python3
"""
测试所有场景下的基准策略，并生成汇总报告
"""

import os
import sys
import json
import time
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从baseline_strategies.py导入环境
from Scripts.Chapter5.Env_Ultra import EnvUltra
from Scripts.utils.global_utils import font_get

# 设置字体
font_get()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 全局字体大小设置
FONT_SIZE = 24

def run_strategy_test(scenario, strategy, output_base_dir):
    """
    运行单个策略测试
    
    Args:
        scenario: 场景类型
        strategy: 策略类型
        output_base_dir: 输出基础目录
    
    Returns:
        tuple: (scenario, strategy, test_result_path, power_svg_path)
    """
    print(f"🚀 开始测试: {scenario} - {strategy}")
    
    # 运行测试命令
    cmd = [
        sys.executable,
        os.path.join(current_file_dir, "baseline_strategies.py"),
        "--scenario", scenario,
        "--strategy", strategy,
        "--output-dir", output_base_dir
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 测试完成: {scenario} - {strategy}")
        
        # 返回结果文件路径
        test_result_path = os.path.join(output_base_dir, f"test_result_{scenario}_{strategy}.json")
        power_svg_path = os.path.join(output_base_dir, f"power_distribution_{scenario}_{strategy}.svg")
        
        return (scenario, strategy, test_result_path, power_svg_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {scenario} - {strategy}")
        print(f"   错误信息: {e.stderr}")
        return (scenario, strategy, None, None)

def generate_summary_report(results, output_dir):
    """
    生成汇总报告
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    print("\n📊 生成汇总报告...")
    
    # 汇总数据
    summary_data = {
        'total_tests': len(results),
        'successful_tests': sum(1 for r in results if r[2] is not None),
        'test_results': [],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存每个测试的结果
    all_rewards = {}
    for scenario, strategy, result_path, svg_path in results:
        if result_path and os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            summary_data['test_results'].append({
                'scenario': scenario,
                'strategy': strategy,
                'total_reward': result['total_reward'],
                'total_steps': result['total_steps'],
                'average_reward_per_step': result['average_reward_per_step'],
                'result_path': result_path,
                'svg_path': svg_path
            })
            
            # 按策略和场景保存奖励
            if strategy not in all_rewards:
                all_rewards[strategy] = {}
            all_rewards[strategy][scenario] = result['total_reward']
    
    # 保存汇总结果到JSON文件
    summary_json_path = os.path.join(output_dir, "summary_report.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 汇总报告已保存到: {summary_json_path}")
    
    # 生成奖励比较图表
    generate_reward_comparison_chart(all_rewards, output_dir)
    
    # 生成功率分配汇总图
    generate_power_summary_plots(results, output_dir)
    
    return summary_json_path

def generate_reward_comparison_chart(all_rewards, output_dir):
    """
    生成不同策略在不同场景下的奖励比较图表
    
    Args:
        all_rewards: 所有策略的奖励数据
        output_dir: 输出目录
    """
    print("📈 生成奖励比较图表...")
    
    # 整理数据
    strategies = list(all_rewards.keys())
    scenarios = list(all_rewards[strategies[0]].keys())
    
    # 准备图表数据
    x = np.arange(len(scenarios))
    width = 0.35  # 柱状图宽度
    
    # 创建图表
    plt.figure(figsize=(14, 6))
    
    # 为每个策略绘制柱状图
    for i, strategy in enumerate(strategies):
        rewards = [all_rewards[strategy][scenario] for scenario in scenarios]
        plt.bar(x + i * width, rewards, width, label=strategy)
    
    # 添加标签和标题
    plt.xlabel('Scenario', fontsize=FONT_SIZE, fontweight='bold')
    plt.ylabel('Total Reward', fontsize=FONT_SIZE, fontweight='bold')
    plt.title('Comparison of Baseline Strategies Across Scenarios', fontsize=FONT_SIZE, fontweight='bold')
    plt.xticks(x + width/2, scenarios, rotation=45, ha='right', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(output_dir, "reward_comparison_chart.svg")
    plt.savefig(chart_path, dpi=1200, bbox_inches='tight')
    
    chart_png_path = os.path.join(output_dir, "reward_comparison_chart.png")
    plt.savefig(chart_png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 奖励比较图表已保存到:")
    print(f"   SVG: {chart_path}")
    print(f"   PNG: {chart_png_path}")
    
    # 关闭图表
    plt.close()


def generate_power_summary_plots(results, output_dir):
    """
    生成功率分配汇总图
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    print("📈 生成功率分配汇总图...")
    
    # 整理结果数据
    power_data_dict = {}
    for scenario, strategy, result_path, svg_path in results:
        if result_path and os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                power_data_dict[(scenario, strategy)] = result_data
    
    # 获取所有策略
    strategies = list(set(strategy for _, strategy, _, _ in results if _[2] is not None))
    
    # 1. 生成9种基础环境的功率分配汇总图（3x3子图）
    # 获取9种基础环境
    base_scenarios = ['air', 'surface', 'underwater', 
                     'air_to_surface', 'surface_to_air', 
                     'air_to_underwater', 'underwater_to_air', 
                     'surface_to_underwater', 'underwater_to_surface']
    
    if all((scenario, strategies[0]) in power_data_dict for scenario in base_scenarios):
        generate_9_scenarios_power_plot(base_scenarios, strategies[0], power_data_dict, output_dir)
    
    # 2. 生成3种典型剖面的功率分配汇总图（3x1子图）
    typical_scenarios = ['cruise', 'recon', 'rescue']
    if all((scenario, strategies[0]) in power_data_dict for scenario in typical_scenarios):
        generate_typical_scenarios_power_plot(typical_scenarios, strategies[0], power_data_dict, output_dir)


def generate_9_scenarios_power_plot(scenarios, strategy, power_data_dict, output_dir):
    """
    生成9种基础环境的功率分配汇总图
    
    Args:
        scenarios: 9种基础环境列表
        strategy: 策略类型
        power_data_dict: 功率数据字典
        output_dir: 输出目录
    """
    # 创建3x3子图，增加宽度以留出更多坐标轴空间
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharex=True, sharey=True)
    # 设置子图之间的间距
    fig.subplots_adjust(left=0.04, right=0.94, top=0.92, bottom=0.12, wspace=0.6, hspace=0.3)
    fig.suptitle(f'Power Distribution for 9 Basic Scenarios Rule Based Strategy', fontsize=FONT_SIZE, fontweight='bold', y=0.98)
    
    # 颜色配置
    colors = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    
    # 模态背景色映射
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
    
    # 绘制每个子图
    for i, scenario in enumerate(scenarios):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 获取数据
        data_key = (scenario, strategy)
        if data_key in power_data_dict:
            power_data = power_data_dict[data_key]['power_data']
            times = np.arange(len(power_data['load_power']))
            
            # 绘制功率曲线
            ax.plot(times, power_data['load_power'], label='Power Demand', color=colors[0], linewidth=1.5)
            ax.plot(times, power_data['power_fc'], label='Power Fuel Cell', color=colors[1], linewidth=1.5)
            ax.plot(times, power_data['power_bat'], label='Power Battery', color=colors[2], linewidth=1.5)
            ax.plot(times, power_data['power_sc'], label='Power SuperCap', color='k', linestyle='--', linewidth=1.5)
            
            # 添加SOC曲线（右轴1）
            ax2 = ax.twinx()
            ax2.plot(times, power_data['soc_bat'], label='Battery SOC', color=colors[3], alpha=0.7, linewidth=1.0)
            ax2.plot(times, power_data['soc_sc'], label='SuperCap SOC', color='grey', linestyle=':', alpha=0.7, linewidth=1.0)
            ax2.set_ylabel('SOC', fontsize=FONT_SIZE)
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='y', labelsize=FONT_SIZE)
            
            # 添加温度曲线（右轴2，向外偏移）
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 80))  # 增加向外偏移距离到80
            ax3.plot(times, power_data['temperature'], label='Environment Temperature', color=colors[4], alpha=0.7, linewidth=1.0)
            ax3.set_ylabel('Temperature/°C', color=colors[4], fontsize=FONT_SIZE)
            ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=FONT_SIZE)
            ax3.set_ylim(-25, 40)
            
            # 配置子图
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=FONT_SIZE, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(-2500, 5500)
            
            # 只在最后一行添加x轴标签
            if row == 2:
                ax.set_xlabel('Time/s', fontsize=FONT_SIZE)
            
            # 只在第一列添加y轴标签
            if col == 0:
                ax.set_ylabel('Power/W', fontsize=FONT_SIZE)
    
    # 统一添加图例
    fig.legend(['Power Demand', 'Power Fuel Cell', 'Power Battery', 'Power SuperCap', 
               'Battery SOC', 'SuperCap SOC', 'Environment Temperature'], 
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=FONT_SIZE)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图表
    svg_path = os.path.join(output_dir, f"power_distribution_9_base_scenarios_{strategy}.svg")
    png_path = os.path.join(output_dir, f"power_distribution_9_base_scenarios_{strategy}.png")
    
    plt.savefig(svg_path, bbox_inches='tight', dpi=1200)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 9种基础环境功率分配汇总图已保存到:")
    print(f"   SVG: {svg_path}")
    print(f"   PNG: {png_path}")
    
    plt.close()


def generate_typical_scenarios_power_plot(scenarios, strategy, power_data_dict, output_dir):
    """
    生成3种典型剖面的功率分配汇总图
    
    Args:
        scenarios: 3种典型剖面列表
        strategy: 策略类型
        power_data_dict: 功率数据字典
        output_dir: 输出目录
    """
    # 创建3x1子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    fig.suptitle(f'Power Distribution for 3 Typical Profiles Rule Based  Strategy', fontsize=FONT_SIZE, fontweight='bold', y=0.98)
    
    # 颜色配置
    colors = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    
    # 模态背景色映射
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
    
    # 绘制每个子图
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        
        # 获取数据
        data_key = (scenario, strategy)
        if data_key in power_data_dict:
            result_data = power_data_dict[data_key]
            power_data = result_data['power_data']
            times = np.arange(len(power_data['load_power']))
            
            # 绘制功率曲线
            l1, = ax.plot(times, power_data['load_power'], label='Power Demand', color=colors[0], linewidth=2)
            l2, = ax.plot(times, power_data['power_fc'], label='Power Fuel Cell', color=colors[1], linewidth=2)
            l3, = ax.plot(times, power_data['power_bat'], label='Power Battery', color=colors[2], linewidth=2)
            l4, = ax.plot(times, power_data['power_sc'], label='Power SuperCap', color='k', linestyle='--', linewidth=2)
            
            # 配置子图
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=FONT_SIZE, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(-2500, 5500)
            ax.set_ylabel('Power/W', fontsize=FONT_SIZE)
            ax.tick_params(axis='both', labelsize=FONT_SIZE)
            
            # 为所有子图添加SOC曲线（右轴1）
            ax2 = ax.twinx()
            ax2.plot(times, power_data['soc_bat'], label='Battery SOC', color=colors[3], alpha=0.7, linewidth=1.5)
            ax2.plot(times, power_data['soc_sc'], label='SuperCap SOC', color='grey', linestyle=':', alpha=0.7, linewidth=1.5)
            ax2.set_ylabel('SOC', fontsize=FONT_SIZE)
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='y', labelsize=FONT_SIZE)
            
            # 为所有子图添加温度曲线（右轴2，向外偏移）
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 65))  # 向外偏移65
            ax3.plot(times, power_data['temperature'], label='Environment Temperature', color=colors[4], alpha=0.7, linewidth=1.5)
            ax3.set_ylabel('Environment Temperature/°C', color=colors[4], fontsize=FONT_SIZE)
            ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=FONT_SIZE)
            ax3.set_ylim(-25, 40)
    
    # 添加统一的x轴标签
    axes[-1].set_xlabel('Time/s', fontsize=FONT_SIZE)
    axes[-1].tick_params(axis='x', labelsize=FONT_SIZE)
    
    # 统一添加图例
    fig.legend(['Power Demand', 'Power Fuel Cell', 'Power Battery', 'Power SuperCap', 
               'Battery SOC', 'SuperCap SOC', 'Environment Temperature'], 
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=FONT_SIZE)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图表
    svg_path = os.path.join(output_dir, f"power_distribution_3_typical_scenarios_{strategy}.svg")
    png_path = os.path.join(output_dir, f"power_distribution_3_typical_scenarios_{strategy}.png")
    
    plt.savefig(svg_path, bbox_inches='tight', dpi=1200)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 3种典型剖面功率分配汇总图已保存到:")
    print(f"   SVG: {svg_path}")
    print(f"   PNG: {png_path}")
    
    plt.close()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='测试所有场景下的基准策略')
    parser.add_argument('--output-dir', type=str, default='', 
                        help='输出目录')
    parser.add_argument('--parallel', action='store_true', 
                        help='是否并行测试')
    parser.add_argument('--max-workers', type=int, default=4, 
                        help='并行测试的最大线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not args.output_dir:
        timestamp = time.strftime("%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "nets", "Chap5", "baseline_results", f"all_scenarios_{timestamp}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 开始测试所有场景下的基准策略")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 获取所有场景类型
    all_scenarios = EnvUltra.SCENARIO_TYPES
    all_strategies = ['rule_based']
    
    print(f"\n📋 测试计划:")
    print(f"场景数量: {len(all_scenarios)}")
    print(f"策略数量: {len(all_strategies)}")
    print(f"总测试数: {len(all_scenarios) * len(all_strategies)}")
    print(f"并行测试: {'是' if args.parallel else '否'}")
    if args.parallel:
        print(f"最大线程数: {args.max_workers}")
    
    print(f"\n场景列表: {', '.join(all_scenarios)}")
    print(f"策略列表: {', '.join(all_strategies)}")
    
    start_time = time.time()
    
    # 运行测试
    results = []
    if args.parallel:
        # 并行测试
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有测试任务
            futures = []
            for scenario in all_scenarios:
                for strategy in all_strategies:
                    futures.append(executor.submit(run_strategy_test, scenario, strategy, output_dir))
            
            # 收集结果
            for future in futures:
                results.append(future.result())
    else:
        # 串行测试
        for scenario in all_scenarios:
            for strategy in all_strategies:
                result = run_strategy_test(scenario, strategy, output_dir)
                results.append(result)
    
    end_time = time.time()
    
    print(f"\n=" * 60)
    print(f"测试完成！总耗时: {end_time - start_time:.2f} 秒")
    print("=" * 60)
    
    # 生成汇总报告
    generate_summary_report(results, output_dir)
    
    print(f"\n🎉 所有测试结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
