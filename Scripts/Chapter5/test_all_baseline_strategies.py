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
    plt.xlabel('Scenario', fontsize=14, fontweight='bold')
    plt.ylabel('Total Reward', fontsize=14, fontweight='bold')
    plt.title('Comparison of Baseline Strategies Across Scenarios', fontsize=16, fontweight='bold')
    plt.xticks(x + width/2, scenarios, rotation=45, ha='right', fontsize=11)
    plt.legend(fontsize=12)
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
    all_strategies = ['rule_based', 'dp']
    
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
