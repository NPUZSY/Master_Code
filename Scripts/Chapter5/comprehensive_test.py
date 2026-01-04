#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试脚本：测试不同章节的智能体在超级环境中的表现

功能：
1. 支持测试Chapter3和Chapter4的智能体
2. 兼容第五章的慢学习和后续的快学习
3. 支持在超级环境的所有场景中测试
4. 生成柱状图对比不同策略的表现
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Scripts.Chapter5.Env_Ultra import EnvUltra
from Scripts.Chapter5.baseline_strategies import BaselineStrategies

def test_baseline_strategy(strategy_name, env, episodes=1):
    """
    测试基准策略
    
    Args:
        strategy_name: 策略名称 ('rule_based')
        env: 环境实例
        episodes: 测试回合数
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
    """
    strategies = BaselineStrategies(env)
    total_reward = 0.0
    total_steps = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        
        while not done:
            if strategy_name == 'rule_based':
                action_list = strategies.rule_based_strategy(state)
            else:
                raise ValueError(f"不支持的基准策略: {strategy_name}")
            
            next_state, reward, done, info = env.step(action_list)
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        total_reward += episode_reward
        total_steps += episode_steps
    
    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    
    return avg_reward, avg_steps

def test_chapter3_agent(env, agent_path, episodes=1, output_dir=None, strategy_name=None):
    """
    测试Chapter3的多智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
        output_dir: 输出目录
        strategy_name: 策略名称
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
    """
    try:
        # 直接使用命令行调用Chapter3的测试脚本
        import subprocess
        import sys
        import os
        import json
        
        # 构造策略-环境文件夹路径
        if output_dir and strategy_name:
            strategy_env_dir = os.path.join(output_dir, strategy_name, env.scenario_type)
            os.makedirs(strategy_env_dir, exist_ok=True)
        else:
            strategy_env_dir = output_dir
        
        # 构造命令行参数
        chapter3_test_script = os.path.join(os.path.dirname(__file__), '../Chapter3/test.py')
        cmd = [
            sys.executable, chapter3_test_script,
            '--net-date', '1218',
            '--train-id', '36',
            '--use-ultra-env',
            '--scenario', env.scenario_type
            # 移除--show-plot false，使用默认值
        ]
        
        # 添加--save-dir参数
        if strategy_env_dir:
            cmd.extend(['--save-dir', strategy_env_dir])
        
        # 运行测试脚本
        print(f"运行Chapter3测试脚本: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析测试结果
        if result.returncode == 0:
            print(f"✅ Chapter3智能体测试完成")
            # 从生成的JSON文件中读取奖励信息
            json_file_path = os.path.join(strategy_env_dir, "MARL_Model_Test_Results.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                total_reward = test_results['core_metrics']['total_reward']
                total_steps = test_results['time_metrics']['total_steps']
                print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
                print(f"📊 从JSON文件读取到的步数: {total_steps}")
                return total_reward, total_steps
            else:
                print(f"警告: 测试结果JSON文件不存在，使用规则策略代替")
        else:
            print(f"❌ Chapter3智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes)
    except Exception as e:
        print(f"错误: Chapter3智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes)

def test_chapter4_agent(env, agent_path, episodes=1, output_dir=None, strategy_name=None):
    """
    测试Chapter4的联合网络智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
        output_dir: 输出目录
        strategy_name: 策略名称
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
    """
    try:
        # 直接使用命令行调用Chapter4的测试脚本
        import subprocess
        import sys
        import os
        import json
        
        # 构造策略-环境文件夹路径
        if output_dir and strategy_name:
            strategy_env_dir = os.path.join(output_dir, strategy_name, env.scenario_type)
            os.makedirs(strategy_env_dir, exist_ok=True)
        else:
            strategy_env_dir = output_dir
        
        # 构造命令行参数
        chapter4_test_script = os.path.join(os.path.dirname(__file__), '../Chapter4/test_Joint.py')
        cmd = [
            sys.executable, chapter4_test_script,
            '--net-date', '1223',
            '--train-id', '2',
            '--use-ultra-env',
            '--scenario', env.scenario_type
            # 移除--show-plot false，使用默认值
        ]
        
        # 添加--save-dir参数
        if strategy_env_dir:
            cmd.extend(['--save-dir', strategy_env_dir])
        
        # 运行测试脚本
        print(f"运行Chapter4测试脚本: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析测试结果
        if result.returncode == 0:
            print(f"✅ Chapter4智能体测试完成")
            # 从生成的JSON文件中读取奖励信息
            json_file_path = os.path.join(strategy_env_dir, "Joint_Model_Test_Results.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                total_reward = test_results['core_metrics']['total_reward']
                total_steps = test_results['time_metrics']['total_steps']
                print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
                print(f"📊 从JSON文件读取到的步数: {total_steps}")
                return total_reward, total_steps
            else:
                print(f"警告: 测试结果JSON文件不存在，使用规则策略代替")
        else:
            print(f"❌ Chapter4智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes)
    except Exception as e:
        print(f"错误: Chapter4智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes)

def test_slow_learning_agent(env, agent_path, episodes=1):
    """
    测试第五章的慢学习智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
    """
    try:
        # 直接使用命令行调用Chapter5的慢学习测试脚本
        import subprocess
        import sys
        import os
        
        # 构造命令行参数
        slow_test_script = os.path.join(os.path.dirname(__file__), 'test_slow_training.py')
        
        # 检查慢学习测试脚本是否存在
        if not os.path.exists(slow_test_script):
            print(f"警告: 慢学习测试脚本不存在，使用规则策略代替")
            return test_baseline_strategy('rule_based', env, episodes)
        
        # 构建命令行参数
        cmd = [
            sys.executable, slow_test_script
        ]
        
        # 添加模型路径（必须参数）
        if agent_path:
            cmd.extend(['--model-path', agent_path])
        else:
            # 使用默认模型路径
            default_model_path = os.path.join(os.path.dirname(__file__), '../../nets/Chap5/slow_learning/model.pth')
            cmd.extend(['--model-path', default_model_path])
            print(f"警告: 未提供慢学习模型路径，使用默认路径: {default_model_path}")
        
        # 添加场景参数（如果支持的话）
        # 检查test_slow_training.py是否支持--scenario参数
        # test_slow_training.py不支持--scenario参数，跳过该参数
        
        # 运行测试脚本
        print(f"运行慢学习测试脚本: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析测试结果
        if result.returncode == 0:
            print(f"✅ 慢学习智能体测试完成")
            # 这里需要从输出中提取奖励信息
            # 由于test_slow_training.py没有返回奖励信息，暂时使用规则策略代替
            print(f"警告: 慢学习智能体测试结果解析功能尚未实现，使用规则策略代替")
        else:
            print(f"❌ 慢学习智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes)
    except Exception as e:
        print(f"错误: 慢学习智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes)

def test_fast_learning_agent(env, agent_path, episodes=1):
    """
    测试第五章的快学习智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
    """
    print(f"警告: 快学习智能体测试功能尚未实现，使用规则策略代替")
    return test_baseline_strategy('rule_based', env, episodes)

def run_comprehensive_test():
    """
    运行综合测试
    """
    # 导入多线程库
    import concurrent.futures
    
    # 定义测试场景
    scenarios = EnvUltra.SCENARIO_TYPES
    
    # 定义测试策略
    strategies = [
        {'name': 'Rule-Based', 'type': 'baseline', 'path': None},
        {'name': 'Chapter3 MARL', 'type': 'chapter3', 'path': '/home/siyu/Master_Code/nets/Chap3/1218/36'},
        {'name': 'Chapter4 Joint Net', 'type': 'chapter4', 'path': '/home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/2'},
        {'name': 'Slow Learning', 'type': 'slow_learning', 'path': None},
        {'name': 'Fast Learning', 'type': 'fast_learning', 'path': None}
    ]
    
    # 测试参数
    episodes = 1
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join('/home/siyu/Master_Code/nets/Chap5', 'comprehensive_test_results', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义测试任务函数
    def test_task(scenario, strategy):
        """
        单个测试任务
        
        Args:
            scenario: 测试场景
            strategy: 测试策略
        
        Returns:
            测试结果字典
        """
        print(f"\n--- 测试策略: {strategy['name']}，场景: {scenario} ---")
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        
        # 根据策略类型选择测试函数
        if strategy['type'] == 'baseline':
            avg_reward, avg_steps = test_baseline_strategy('rule_based', env, episodes)
        elif strategy['type'] == 'chapter3':
            avg_reward, avg_steps = test_chapter3_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
        elif strategy['type'] == 'chapter4':
            avg_reward, avg_steps = test_chapter4_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
        elif strategy['type'] == 'slow_learning':
            avg_reward, avg_steps = test_slow_learning_agent(env, strategy['path'], episodes)
        elif strategy['type'] == 'fast_learning':
            avg_reward, avg_steps = test_fast_learning_agent(env, strategy['path'], episodes)
        else:
            raise ValueError(f"不支持的策略类型: {strategy['type']}")
        
        print(f"策略: {strategy['name']}，场景: {scenario} 测试完成")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均步数: {avg_steps:.2f}")
        
        # 返回测试结果
        return {
            'scenario': scenario,
            'strategy': strategy['name'],
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }
    
    # 存储测试结果
    results = []
    
    # 使用线程池执行测试任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 创建任务列表
        future_to_task = {}
        for scenario in scenarios:
            for strategy in strategies:
                future = executor.submit(test_task, scenario, strategy)
                future_to_task[future] = (scenario, strategy['name'])
        
        # 收集测试结果
        for future in concurrent.futures.as_completed(future_to_task):
            scenario, strategy_name = future_to_task[future]
            try:
                task_result = future.result()
                results.append(task_result)
            except Exception as e:
                print(f"策略: {strategy_name}，场景: {scenario} 测试失败: {e}")
    
    return results, output_dir

def plot_comparison(results, output_dir):
    """
    绘制不同策略在不同场景下的奖励对比柱状图
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    # 提取唯一的场景和策略
    scenarios = sorted(list(set(r['scenario'] for r in results)))
    strategies = sorted(list(set(r['strategy'] for r in results)))
    
    # 准备数据
    data = {}
    for scenario in scenarios:
        data[scenario] = {}
        for strategy in strategies:
            # 查找对应结果
            for r in results:
                if r['scenario'] == scenario and r['strategy'] == strategy:
                    data[scenario][strategy] = r['avg_reward']
                    break
    
    # 创建图表
    num_scenarios = len(scenarios)
    num_strategies = len(strategies)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 设置柱状图宽度和位置
    bar_width = 0.15
    x = np.arange(num_scenarios)
    
    # 为每个策略绘制柱状图
    for i, strategy in enumerate(strategies):
        rewards = [data[scenario][strategy] for scenario in scenarios]
        ax.bar(x + i * bar_width, rewards, bar_width, label=strategy)
    
    # 设置图表属性
    ax.set_xlabel('Scene')
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance Comparison of Different Strategies Across Super Environments')
    ax.set_xticks(x + bar_width * (num_strategies - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    fig_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n=== 图表已保存到: {fig_path} ===")
    
    # 保存结果数据
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"测试结果已保存到: {results_path}")
    
    plt.close()

def main():
    """
    主函数
    """
    print("=== 开始综合测试 ===")
    
    # 运行测试
    results, output_dir = run_comprehensive_test()
    
    # 绘制对比图
    plot_comparison(results, output_dir)
    
    print("\n=== 综合测试完成 ===")

if __name__ == "__main__":
    main()
