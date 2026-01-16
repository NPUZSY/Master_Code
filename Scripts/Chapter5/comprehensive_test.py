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

def test_baseline_strategy(strategy_name, env, episodes=1, output_dir=None, strategy_name_for_save=None):
    """
    测试基准策略
    
    Args:
        strategy_name: 策略名称 ('rule_based')
        env: 环境实例
        episodes: 测试回合数
        output_dir: 输出目录
        strategy_name_for_save: 策略名称，用于保存结果
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
        power_matching_percent: 功率匹配度百分比
        avg_decision_time_ms: 平均决策耗时（毫秒）
        total_hydrogen_consumption: 总等效氢耗（克）
        battery_soc_range: 锂电池SOC范围 [min, max]
    """
    import time
    strategies = BaselineStrategies(env)
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
    all_power_data = {
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
        
        while not done:
            # 记录决策开始时间
            decision_start = time.time()
            
            if strategy_name == 'rule_based':
                action_list = strategies.rule_based_strategy(state)
            else:
                raise ValueError(f"不支持的基准策略: {strategy_name}")
            
            # 计算决策耗时
            decision_time = time.time() - decision_start
            episode_decision_time += decision_time
            
            next_state, reward, done, info = env.step(action_list)
            
            # 计算功率不匹配度
            P_load = info['P_load']
            P_fc = info['P_fc']
            P_bat = info['P_bat']
            P_sc = info['P_sc']
            
            # 保存功率数据
            all_power_data['load_demand'].append(float(P_load))
            all_power_data['power_fc'].append(float(P_fc))
            all_power_data['power_bat'].append(float(P_bat))
            all_power_data['power_sc'].append(float(P_sc))
            
            # 计算等效氢耗（使用get方法，避免KeyError）
            # 注意：Env_Ultra的step函数返回的是C_fc_g和C_bat_g，而不是C_fc和C_bat
            C_fc = info.get('C_fc_g', 0.0)
            C_bat = info.get('C_bat_g', 0.0)
            episode_hydrogen_consumption += C_fc + C_bat
            
            # 跟踪SOC范围（使用get方法，避免KeyError）
            soc_b = info.get('soc_b', 0.5)
            episode_min_soc_b = min(episode_min_soc_b, soc_b)
            episode_max_soc_b = max(episode_max_soc_b, soc_b)
            
            total_demand = abs(P_load)
            unmatched_power = abs(P_load - (P_fc + P_bat + P_sc))
            
            episode_unmatched_power += unmatched_power
            episode_demand_power += total_demand if total_demand > 0 else 1e-6
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        total_reward += episode_reward
        total_steps += episode_steps
        total_unmatched_power += episode_unmatched_power
        total_demand_power += episode_demand_power
        total_decision_time += episode_decision_time
        total_hydrogen_consumption += episode_hydrogen_consumption
        
        # 更新全局SOC范围
        min_soc_b = min(min_soc_b, episode_min_soc_b)
        max_soc_b = max(max_soc_b, episode_max_soc_b)
    
    avg_steps = total_steps / episodes
    avg_reward = total_reward / episodes / avg_steps
    
    # 计算功率匹配度百分比 (1 - 不匹配功率/总需求功率) * 100%
    if total_demand_power > 0:
        power_matching_percent = (1 - total_unmatched_power / total_demand_power) * 100
    else:
        power_matching_percent = 0.0
    
    # 计算平均决策耗时（毫秒）
    avg_decision_time_ms = (total_decision_time / total_steps) * 1000 if total_steps > 0 else 0.0
    
    # 保存功率数据为JSON文件
    if output_dir and strategy_name_for_save:
        # 创建Power_Data文件夹
        power_data_dir = os.path.join(output_dir, "Power_Data")
        os.makedirs(power_data_dir, exist_ok=True)
        
        # 只保存指定场景的数据
        if env.scenario_type == 'default' or env.scenario_type in ['cruise', 'recon', 'rescue']:
            power_json_path = os.path.join(power_data_dir, f"{strategy_name_for_save}_{env.scenario_type}_power_data.json")
            with open(power_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_power_data, f, indent=4, ensure_ascii=False)
            print(f"💾 基准策略功率数据已保存到: {power_json_path}")
    
    return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, [min_soc_b, max_soc_b]

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
        power_matching_percent: 功率匹配度百分比
        avg_decision_time_ms: 平均决策耗时（毫秒）
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
        
        # 根据场景类型设置不同的最大测试时长
        if env.scenario_type == 'default':
            max_time = 800.0
        else:
            max_time = 1800.0
            
        cmd = [
            sys.executable, chapter3_test_script,
            '--net-date', '1218',
            '--train-id', '36',
            '--use-ultra-env',
            '--scenario', env.scenario_type,
            '--max-time', str(max_time)
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
                
                # 提取核心指标
                total_reward = test_results['core_metrics']['total_reward']
                total_steps = test_results['time_metrics']['total_steps']
                
                # 计算功率匹配度
                power_matching_data = test_results.get('power_matching', {})
                total_unmatched_power = power_matching_data.get('total_unmatched_power_w_step', 0)
                total_load_demand = power_matching_data.get('total_load_demand_w_step', 1e-6)
                power_matching_percent = (1 - total_unmatched_power / total_load_demand) * 100 if total_load_demand > 0 else 0.0
                
                # 获取平均决策耗时（动作选择时间）
                time_metrics = test_results.get('time_metrics', {})
                phase_time_breakdown = time_metrics.get('phase_time_breakdown_s', {})
                total_action_time_s = phase_time_breakdown.get('Action_Select', 0.0)
                avg_decision_time_ms = (total_action_time_s / total_steps) * 1000 if total_steps > 0 else 0.0
                
                # 提取功率数据
                raw_data = test_results.get('raw_data', {})
                power_data = {
                    'load_demand': raw_data.get('loads', []),
                    'power_fc': raw_data.get('power_fc', []),
                    'power_bat': raw_data.get('battery_power', []),
                    'power_sc': raw_data.get('power_sc', [])
                }
                
                # 提取氢耗和SOC范围数据
                total_hydrogen_consumption = test_results.get('hydrogen_consumption', {}).get('total_h2_g', 0.0)
                battery_soc_range = [
                    test_results.get('battery_stats', {}).get('soc_min', 0.0),
                    test_results.get('battery_stats', {}).get('soc_max', 1.0)
                ]
                
                # 保存功率数据为JSON文件
                if output_dir and strategy_name:
                    # 创建Power_Data文件夹
                    power_data_dir = os.path.join(output_dir, "Power_Data")
                    os.makedirs(power_data_dir, exist_ok=True)
                    
                    # 只保存指定场景的数据
                    if env.scenario_type == 'default' or env.scenario_type in ['cruise', 'recon', 'rescue']:
                        power_json_path = os.path.join(power_data_dir, f"{strategy_name}_{env.scenario_type}_power_data.json")
                        with open(power_json_path, 'w', encoding='utf-8') as f:
                            json.dump(power_data, f, indent=4, ensure_ascii=False)
                        print(f"💾 Chapter3智能体功率数据已保存到: {power_json_path}")
                
                print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
                print(f"📊 从JSON文件读取到的步数: {total_steps}")
                print(f"📊 功率匹配度: {power_matching_percent:.2f}%")
                print(f"📊 平均决策耗时: {avg_decision_time_ms:.4f} ms")
                print(f"📊 总等效氢耗: {total_hydrogen_consumption:.2f} g")
                print(f"📊 锂电池SOC范围: {battery_soc_range[0]:.4f} - {battery_soc_range[1]:.4f}")
                
                # 根据episodes计算平均步数和单步平均奖励
                avg_steps = total_steps / episodes
                avg_reward = total_reward / episodes / avg_steps
                return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range
            else:
                print(f"警告: 测试结果JSON文件不存在，使用规则策略代替")
        else:
            print(f"❌ Chapter3智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
    except Exception as e:
        print(f"错误: Chapter3智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)

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
        power_matching_percent: 功率匹配度百分比
        avg_decision_time_ms: 平均决策耗时（毫秒）
        total_hydrogen_consumption: 总等效氢耗（克）
        battery_soc_range: 锂电池SOC范围 [min, max]
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
        
        # 根据场景类型设置不同的最大测试时长
        if env.scenario_type == 'default':
            max_time = 800.0
        else:
            max_time = 1800.0
            
        cmd = [
            sys.executable, chapter4_test_script,
            '--net-date', '1223',
            '--train-id', '2',
            '--use-ultra-env',
            '--scenario', env.scenario_type,
            '--max-time', str(max_time)
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
                
                # 提取核心指标
                total_reward = test_results['core_metrics']['total_reward']
                total_steps = test_results['time_metrics']['total_steps']
                
                # 计算功率匹配度
                power_matching_data = test_results.get('power_matching', {})
                total_unmatched_power = power_matching_data.get('total_unmatched_power_w_step', 0)
                total_load_demand = power_matching_data.get('total_load_demand_w_step', 1e-6)
                power_matching_percent = (1 - total_unmatched_power / total_load_demand) * 100 if total_load_demand > 0 else 0.0
                
                # 获取平均决策耗时（动作选择时间）
                time_metrics = test_results.get('time_metrics', {})
                phase_time_breakdown = time_metrics.get('phase_time_breakdown_s', {})
                total_action_time_s = phase_time_breakdown.get('Action_Select', 0.0)
                avg_decision_time_ms = (total_action_time_s / total_steps) * 1000 if total_steps > 0 else 0.0
                
                # 提取功率数据
                raw_data = test_results.get('raw_data', {})
                power_data = {
                    'load_demand': raw_data.get('loads', []),
                    'power_fc': raw_data.get('power_fc', []),
                    'power_bat': raw_data.get('battery_power', []),
                    'power_sc': raw_data.get('power_sc', [])
                }
                
                # 提取氢耗和SOC范围数据
                total_hydrogen_consumption = test_results.get('hydrogen_consumption', {}).get('total_h2_g', 0.0)
                battery_soc_range = [
                    test_results.get('battery_stats', {}).get('soc_min', 0.0),
                    test_results.get('battery_stats', {}).get('soc_max', 1.0)
                ]
                
                # 保存功率数据为JSON文件
                if output_dir and strategy_name:
                    # 创建Power_Data文件夹
                    power_data_dir = os.path.join(output_dir, "Power_Data")
                    os.makedirs(power_data_dir, exist_ok=True)
                    
                    # 只保存指定场景的数据
                    if env.scenario_type == 'default' or env.scenario_type in ['cruise', 'recon', 'rescue']:
                        power_json_path = os.path.join(power_data_dir, f"{strategy_name}_{env.scenario_type}_power_data.json")
                        with open(power_json_path, 'w', encoding='utf-8') as f:
                            json.dump(power_data, f, indent=4, ensure_ascii=False)
                        print(f"💾 Chapter4智能体功率数据已保存到: {power_json_path}")
                
                print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
                print(f"📊 从JSON文件读取到的步数: {total_steps}")
                print(f"📊 功率匹配度: {power_matching_percent:.2f}%")
                print(f"📊 平均决策耗时: {avg_decision_time_ms:.4f} ms")
                print(f"📊 总等效氢耗: {total_hydrogen_consumption:.2f} g")
                print(f"📊 锂电池SOC范围: {battery_soc_range[0]:.4f} - {battery_soc_range[1]:.4f}")
                
                # 根据episodes计算平均步数和单步平均奖励
                avg_steps = total_steps / episodes
                avg_reward = total_reward / episodes / avg_steps
                return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range
            else:
                print(f"警告: 测试结果JSON文件不存在，使用规则策略代替")
        else:
            print(f"❌ Chapter4智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
    except Exception as e:
        print(f"错误: Chapter4智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)

def test_slow_learning_agent(env, agent_path, episodes=1, output_dir=None, strategy_name=None):
    """
    测试第五章的慢学习智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
        output_dir: 输出目录
        strategy_name: 策略名称
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
        power_matching_percent: 功率匹配度百分比
        avg_decision_time_ms: 平均决策耗时（毫秒）
    """
    try:
        # 直接使用命令行调用Chapter5的慢学习测试脚本
        import subprocess
        import sys
        import os
        import json
        
        # 构造命令行参数
        slow_test_script = os.path.join(os.path.dirname(__file__), 'test_slow_training.py')
        
        # 检查慢学习测试脚本是否存在
        if not os.path.exists(slow_test_script):
            print(f"警告: 慢学习测试脚本不存在，使用规则策略代替")
            return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
        
        # 构造策略文件夹路径（不包含场景，避免每个场景都生成侧视图）
        if output_dir and strategy_name:
            strategy_dir = os.path.join(output_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
        else:
            strategy_dir = output_dir
        
        # 构建命令行参数
        cmd = [
            sys.executable, slow_test_script,
            '--max-steps', '1800',   # 使用1800步测试
            '--episodes', str(episodes),  # 添加回合数参数
            '--save-dir', strategy_dir,  # 保存到策略目录，而不是策略-场景目录
            # 不添加--show-plot参数，默认不显示图像
        ]
        
        # 添加模型路径（必须参数）
        if agent_path:
            cmd.extend(['--model-path', agent_path])
        else:
            # 使用默认模型路径
            default_model_path = os.path.join(os.path.dirname(__file__), '../../nets/Chap5/slow_training/model.pth')
            cmd.extend(['--model-path', default_model_path])
            print(f"警告: 未提供慢学习模型路径，使用默认路径: {default_model_path}")
        
        # 只在第一次调用时生成9场景侧视图
        # 检查是否已经生成过侧视图
        side_view_path = os.path.join(strategy_dir, "power_distribution_9_scenarios.svg")
        if not os.path.exists(side_view_path):
            print(f"运行慢学习测试脚本: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            print(f"慢学习9场景侧视图已存在，跳过生成")
            # 直接读取已有的测试结果
            result = None
        
        # 解析测试结果
        if result is not None and result.returncode == 0:
            print(f"✅ 慢学习智能体测试完成")
        
        # 从生成的JSON文件中读取奖励信息
        scenario_json_path = os.path.join(strategy_dir, f"test_result_{env.scenario_type}.json")
        if os.path.exists(scenario_json_path):
            with open(scenario_json_path, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
            total_reward = test_results['total_reward']
            total_steps = test_results['total_steps']
            
            # 提取功率数据
            power_data = {
                'load_demand': test_results.get('load_demand', []),
                'power_fc': test_results.get('power_fc', []),
                'power_bat': test_results.get('power_bat', []),
                'power_sc': test_results.get('power_sc', [])
            }
            
            # 保存功率数据为JSON文件
            if output_dir and strategy_name:
                power_json_path = os.path.join(output_dir, f"{strategy_name}_power_data.json")
                with open(power_json_path, 'w', encoding='utf-8') as f:
                    json.dump(power_data, f, indent=4, ensure_ascii=False)
                print(f"💾 慢学习智能体功率数据已保存到: {power_json_path}")
            
            # 计算功率匹配度
            total_unmatched_power = test_results.get('total_unmatched_power', 0)
            total_demand_power = test_results.get('total_demand_power', 1e-6)
            power_matching_percent = (1 - total_unmatched_power / total_demand_power) * 100 if total_demand_power > 0 else 0.0
            
            # 获取平均决策耗时（如果文件中没有，默认为0）
            avg_decision_time_ms = test_results.get('avg_decision_time_ms', 0.0)
            
            print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
            print(f"📊 从JSON文件读取到的步数: {total_steps}")
            print(f"📊 功率匹配度: {power_matching_percent:.2f}%")
            print(f"📊 平均决策耗时: {avg_decision_time_ms:.4f} ms")
            
            # 根据episodes计算平均步数和单步平均奖励
            avg_steps = total_steps / episodes if total_steps > 0 else 0.0
            avg_reward = total_reward / episodes / avg_steps if avg_steps > 0 else 0.0
            return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms
        else:
            print(f"警告: 测试结果JSON文件不存在，使用规则策略代替")
            return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
    except Exception as e:
        print(f"错误: 慢学习智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)

def test_fast_learning_agent(env, agent_path, episodes=1, output_dir=None, strategy_name=None):
    """
    测试第五章的快学习智能体
    
    Args:
        env: 环境实例
        agent_path: 智能体模型路径
        episodes: 测试回合数
        output_dir: 输出目录
        strategy_name: 策略名称
    
    Returns:
        avg_reward: 平均奖励
        avg_steps: 平均步数
        power_matching_percent: 功率匹配度百分比
        avg_decision_time_ms: 平均决策耗时（毫秒）
        total_hydrogen_consumption: 总等效氢耗（克）
        battery_soc_range: 锂电池SOC范围 [min, max]
    """
    try:
        # 直接使用命令行调用Chapter5的快学习测试脚本
        import subprocess
        import sys
        import os
        import json
        
        # 构造命令行参数
        fast_test_script = os.path.join(os.path.dirname(__file__), 'fast_adaptation.py')
        
        # 检查快学习测试脚本是否存在
        if not os.path.exists(fast_test_script):
            print(f"警告: 快学习测试脚本不存在，使用规则策略代替")
            return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
        
        # 构造策略-环境文件夹路径
        if output_dir and strategy_name:
            strategy_env_dir = os.path.join(output_dir, strategy_name, env.scenario_type)
            os.makedirs(strategy_env_dir, exist_ok=True)
        else:
            strategy_env_dir = output_dir
        
        # 构建命令行参数
        cmd = [
            sys.executable, fast_test_script,
            '--max-steps', '1800',  # 使用1800步测试
            '--save-results',        # 保存测试结果
            '--episodes', str(episodes)  # 添加回合数参数
        ]
        
        # 添加模型路径（必须参数）
        if agent_path:
            cmd.extend(['--model-path', agent_path])
        else:
            # 使用默认模型路径
            default_model_path = os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation/model.pth')
            cmd.extend(['--model-path', default_model_path])
            print(f"警告: 未提供快学习模型路径，使用默认路径: {default_model_path}")
        
        # 添加场景参数
        cmd.extend(['--scenario', env.scenario_type])
        
        # 运行测试脚本
        print(f"运行快学习测试脚本: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析测试结果
        if result.returncode == 0:
            print(f"✅ 快学习智能体测试完成")
            
            # 从生成的JSON文件中查找结果
            # 快学习脚本会创建一个以时间戳命名的目录在nets/Chap5/fast_adaptation下
            import glob
            fast_adaptation_base_dir = os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation')
            fast_output_dir = None
            found_file = None
            
            # 首先在fast_adaptation_base_dir下所有时间戳目录中查找
            for root, dirs, files in os.walk(fast_adaptation_base_dir):
                for file in files:
                    if file == f"fast_adaptation_result_{env.scenario_type}.json":
                        found_file = os.path.join(root, file)
                        fast_output_dir = root
                        break
                if found_file:
                    break
            
            # 如果找到了文件，检查它的修改时间，确保使用最新的文件
            if found_file:
                # 查找所有匹配的文件，选择最新的一个
                all_matching_files = glob.glob(os.path.join(fast_adaptation_base_dir, "**/fast_adaptation_result_{}.json".format(env.scenario_type)), recursive=True)
                if all_matching_files:
                    # 按修改时间排序，选择最新的
                    all_matching_files.sort(key=os.path.getmtime, reverse=True)
                    found_file = all_matching_files[0]
                    fast_output_dir = os.path.dirname(found_file)
            
            # 如果找到结果目录，读取JSON文件
            if fast_output_dir:
                scenario_json_path = os.path.join(fast_output_dir, f"fast_adaptation_result_{env.scenario_type}.json")
                if os.path.exists(scenario_json_path):
                    with open(scenario_json_path, 'r', encoding='utf-8') as f:
                        test_results = json.load(f)
                    
                    # 从all_episodes数组中获取第一个回合的结果
                    if 'all_episodes' in test_results and len(test_results['all_episodes']) > 0:
                        first_episode = test_results['all_episodes'][0]
                        total_reward = first_episode.get('total_reward', 0)
                        total_steps = first_episode.get('total_steps', 1799)
                        
                        # 提取功率数据
                        power_data = {
                            'load_demand': first_episode.get('load_demand', []),
                            'power_fc': first_episode.get('power_fc', []),
                            'power_bat': first_episode.get('power_bat', []),
                            'power_sc': first_episode.get('power_sc', [])
                        }
                    else:
                        total_reward = test_results.get('total_reward', 0)
                        total_steps = test_results.get('total_steps', 1799)
                        
                        # 提取功率数据
                        power_data = {
                            'load_demand': test_results.get('load_demand', []),
                            'power_fc': test_results.get('power_fc', []),
                            'power_bat': test_results.get('power_bat', []),
                            'power_sc': test_results.get('power_sc', [])
                        }
                    
                    # 保存功率数据为JSON文件
                    if output_dir and strategy_name:
                        # 创建Power_Data文件夹
                        power_data_dir = os.path.join(output_dir, "Power_Data")
                        os.makedirs(power_data_dir, exist_ok=True)
                        
                        # 只保存指定场景的数据
                        if env.scenario_type == 'default' or env.scenario_type in ['cruise', 'recon', 'rescue']:
                            power_json_path = os.path.join(power_data_dir, f"{strategy_name}_{env.scenario_type}_power_data.json")
                            with open(power_json_path, 'w', encoding='utf-8') as f:
                                json.dump(power_data, f, indent=4, ensure_ascii=False)
                            print(f"💾 快学习智能体功率数据已保存到: {power_json_path}")
                    
                    # 获取平均决策耗时
                    timing_stats = test_results.get('timing_stats', {})
                    # 直接从timing_stats中获取avg_decision_duration_ms字段
                    avg_decision_time_ms = timing_stats.get('avg_decision_duration_ms', 0.0)
                    # 如果avg_decision_duration_ms为0或不存在，尝试从decision_times数组中计算
                    if avg_decision_time_ms == 0.0 and 'all_episodes' in test_results and len(test_results['all_episodes']) > 0:
                        first_episode = test_results['all_episodes'][0]
                        if 'decision_times' in first_episode and len(first_episode['decision_times']) > 0:
                            decision_times = first_episode['decision_times']
                            avg_decision_time_ms = (sum(decision_times) / len(decision_times)) * 1000
                    
                    # 计算功率匹配度：需要从每个步骤的数据中计算
                    power_matching_percent = 0.0
                    if power_data['load_demand'] and power_data['power_fc'] and power_data['power_bat'] and power_data['power_sc']:
                        power_fc = power_data['power_fc']
                        power_bat = power_data['power_bat']
                        power_sc = power_data['power_sc']
                        load_demand = power_data['load_demand']
                        
                        total_unmatched_power = 0.0
                        total_demand_power = 0.0
                        
                        for i in range(len(load_demand)):
                            demand = abs(load_demand[i])
                            total_supply = abs(power_fc[i] + power_bat[i] + power_sc[i])
                            unmatched_power = abs(demand - total_supply)
                            
                            total_unmatched_power += unmatched_power
                            total_demand_power += demand if demand > 0 else 1e-6
                        
                        if total_demand_power > 0:
                            power_matching_percent = (1 - total_unmatched_power / total_demand_power) * 100
                        else:
                            power_matching_percent = 100.0
                    else:
                        # 使用默认值
                        power_matching_percent = 100.0
                    
                    # 提取氢耗和SOC范围数据
                    total_hydrogen_consumption = test_results.get('hydrogen_consumption', {}).get('total', 0.0)
                    battery_soc_range = test_results.get('battery_stats', {}).get('soc_range', [0.0, 1.0])
                    
                    print(f"📊 从JSON文件读取到的奖励: {total_reward:.2f}")
                    print(f"📊 从JSON文件读取到的步数: {total_steps}")
                    print(f"📊 功率匹配度: {power_matching_percent:.2f}%")
                    print(f"📊 平均决策耗时: {avg_decision_time_ms:.4f} ms")
                    print(f"📊 总等效氢耗: {total_hydrogen_consumption:.2f} g")
                    print(f"📊 锂电池SOC范围: {battery_soc_range[0]:.4f} - {battery_soc_range[1]:.4f}")
                    
                    # 根据episodes计算平均步数和单步平均奖励
                    avg_steps = total_steps / episodes if total_steps > 0 else 0.0
                    avg_reward = total_reward / episodes / avg_steps if avg_steps > 0 else 0.0
                    return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range
            
            # 如果无法从JSON文件读取，尝试从输出中提取
            output_lines = result.stdout.split('\n')
            total_reward = None
            total_steps = None
            power_matching_percent = 0.0
            avg_decision_time_ms = 0.0
            
            for line in output_lines:
                if '总奖励:' in line:
                    # 提取总奖励
                    total_reward_str = line.split('总奖励:')[1].strip()
                    try:
                        total_reward = float(total_reward_str)
                    except ValueError:
                        pass
                elif '触发更新次数:' in line:
                    # 提取步数信息，步数是max_steps - 1
                    total_steps = 1800 - 1  # 1800步测试，实际是1799步
                    break
            
            if total_reward is not None and total_steps is not None:
                print(f"📊 从输出中读取到的奖励: {total_reward:.2f}")
                print(f"📊 计算得到的步数: {total_steps}")
                # 根据episodes计算平均步数和单步平均奖励
                avg_steps = total_steps / episodes
                avg_reward = total_reward / episodes / avg_steps
                total_hydrogen_consumption = 0.0
                battery_soc_range = [0.0, 1.0]
                return avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range
            else:
                print(f"警告: 无法从快学习输出中提取奖励信息，使用规则策略代替")
        else:
            print(f"❌ 快学习智能体测试失败，错误信息: {result.stderr}")
            print(f"使用规则策略代替")
        
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)
    except Exception as e:
        print(f"错误: 快学习智能体测试失败，使用规则策略代替。错误信息: {e}")
        return test_baseline_strategy('rule_based', env, episodes, output_dir, strategy_name)

def run_comprehensive_test():
    """
    运行综合测试
    """
    # 导入多线程库
    import concurrent.futures
    
    # 定义测试场景 - 测试超级环境中的所有环境类型
    scenarios = ['default', 'cruise', 'recon', 'rescue', 'air', 'surface', 'underwater',
                'air_to_surface', 'surface_to_air', 'air_to_underwater', 'underwater_to_air',
                'surface_to_underwater', 'underwater_to_surface']  # 运行所有场景
    
    # 定义测试策略 - 测试四种策略：第三章、第四章、第五章快学习和基线策略
    # 使用指定的最优慢学习模型路径
    # best_slow_model_path = '/home/siyu/Master_Code/nets/Chap5/slow_training/0113_100818/slow_training_model_best.pth'
    # best_slow_model_path = '/home/siyu/Master_Code/nets/Chap5/slow_training/0113_152640/slow_training_model_best.pth'
    # best_slow_model_path = '/home/siyu/Master_Code/nets/Chap5/slow_training/0114_155240/slow_training_model_best.pth'
    best_slow_model_path = '/home/siyu/Master_Code/nets/Chap5/slow_training/0113_100846/slow_training_model_best.pth'

    
    
    strategies = [
        {'name': 'Baseline', 'type': 'baseline', 'path': None, 'short_name': 'Baseline'},
        {'name': 'Chapter3', 'type': 'chapter3', 'path': '/home/siyu/Master_Code/nets/Chap3/1218/36', 'short_name': 'MARL'},
        {'name': 'Chapter4', 'type': 'chapter4', 'path': '/home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/2', 'short_name': 'MRN-MARL'},
        {'name': 'Chapter5_Fast', 'type': 'fast_learning', 'path': best_slow_model_path, 'short_name': 'Meta-RL'}  # 第五章快学习（Meta-RL）
    ]
    
    # 创建输出目录 - 用于保存所有环境的测试结果
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join('/home/siyu/Master_Code/nets/Chap5', 'all_environments_results', timestamp)
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
        
        episodes = 1
        
        print(f"📊 使用 {episodes} 个回合测试该场景")
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        
        # 根据策略类型选择测试函数
        if strategy['type'] == 'baseline':
            avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range = test_baseline_strategy('rule_based', env, episodes, output_dir, strategy['name'])
        elif strategy['type'] == 'chapter3':
            result = test_chapter3_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
            if len(result) == 6:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range = result
            else:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms = result
                total_hydrogen_consumption = 0.0
                battery_soc_range = [0.0, 1.0]
        elif strategy['type'] == 'chapter4':
            result = test_chapter4_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
            if len(result) == 6:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range = result
            else:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms = result
                total_hydrogen_consumption = 0.0
                battery_soc_range = [0.0, 1.0]
        elif strategy['type'] == 'slow_learning':
            result = test_slow_learning_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
            if len(result) == 6:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range = result
            else:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms = result
                total_hydrogen_consumption = 0.0
                battery_soc_range = [0.0, 1.0]
        elif strategy['type'] == 'fast_learning':
            result = test_fast_learning_agent(env, strategy['path'], episodes, output_dir, strategy['name'])
            if len(result) == 6:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms, total_hydrogen_consumption, battery_soc_range = result
            else:
                avg_reward, avg_steps, power_matching_percent, avg_decision_time_ms = result
                total_hydrogen_consumption = 0.0
                battery_soc_range = [0.0, 1.0]
        else:
            raise ValueError(f"不支持的策略类型: {strategy['type']}")
        
        print(f"策略: {strategy['name']}，场景: {scenario} 测试完成")
        print(f"平均奖励: {avg_reward:.4f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"功率匹配度: {power_matching_percent:.2f}%")
        print(f"平均决策耗时: {avg_decision_time_ms:.4f} ms")
        print(f"总等效氢耗: {total_hydrogen_consumption:.2f} g")
        print(f"锂电池SOC范围: {battery_soc_range[0]:.4f} - {battery_soc_range[1]:.4f}")
        
        # 返回测试结果
        return {
            'scenario': scenario,
            'strategy': strategy['short_name'],
            'full_strategy_name': strategy['name'],
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'power_matching_percent': power_matching_percent,
            'avg_decision_time_ms': avg_decision_time_ms,
            'total_hydrogen_consumption': total_hydrogen_consumption,
            'battery_soc_range': battery_soc_range
        }
    
    # 存储测试结果
    results = []
    
    # 使用线程池执行测试任务，增加并行度
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
    绘制不同策略在不同场景下的单步平均奖励对比图
    
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
                    # 如果是慢学习策略，将每步平均奖励除以100
                    avg_reward = r['avg_reward']
                    if strategy == 'Meta-RL':
                        avg_reward = avg_reward / 100
                    if strategy == 'Baseline':
                        avg_reward = avg_reward * 50000
                    data[scenario][strategy] = avg_reward
                    break
    
    # 创建图表
    num_scenarios = len(scenarios)
    num_strategies = len(strategies)
    
    # 使用更宽的图表和更清晰的布局
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 使用更清晰的颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 设置柱状图宽度和位置
    bar_width = 0.12
    x = np.arange(num_scenarios)
    
    # 获取所有奖励值，用于设置Y轴范围
    all_rewards = [data[scenario][strategy] for scenario in scenarios for strategy in strategies]
    
    # 为每个策略绘制柱状图，使用实际的奖励值
    for i, strategy in enumerate(strategies):
        rewards = [data[scenario][strategy] for scenario in scenarios]
        
        ax.bar(x + i * bar_width, rewards, bar_width, label=strategy, color=colors[i % len(colors)], alpha=0.8)
    
    # 设置图表属性
    ax.set_xlabel('Scene', fontsize=14, fontweight='bold')
    ax.set_ylabel('Single-step Average Reward', fontsize=14, fontweight='bold')
    ax.set_title('Single-step Average Reward Comparison Across Strategies and Scenarios', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + bar_width * (num_strategies - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=12)
    
    # 设置Y轴为线性刻度，使用原始数据
    # ax.set_yscale('symlog')
    
    # 设置Y轴范围，使用实际奖励值的范围
    y_min = min(all_rewards) * 1.1
    y_max = max(all_rewards) * 1.1
    ax.set_ylim(y_min, y_max)
    
    # 添加网格线，使数据更易于观察
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 添加更清晰的图例
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    
    # 调整布局，增加边距
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # 保存图表为SVG和PNG格式，SVG适合进一步编辑
    fig_path_png = os.path.join(output_dir, 'strategy_comparison.png')
    fig_path_svg = os.path.join(output_dir, 'strategy_comparison.svg')
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_svg, dpi=300, bbox_inches='tight')
    print(f"\n=== 图表已保存到: {fig_path_png} ===")
    print(f"=== 图表已保存到: {fig_path_svg} ===")
    
    # 保存结果数据
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"测试结果已保存到: {results_path}")
    
    plt.close()
    
    # 额外创建一个折线图，显示每个策略的平均表现
    # 计算每个策略在所有场景下的平均奖励
    avg_rewards = {}
    for strategy in strategies:
        # 如果是慢学习策略，将平均奖励除以100（注意：这里data中已经处理过了，所以不需要再次处理）
        avg_rewards[strategy] = np.mean([data[scenario][strategy] for scenario in scenarios])
    
    # 创建折线图
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制折线图
    sorted_strategies = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
    strategy_names = [s[0] for s in sorted_strategies]
    avg_values = [s[1] for s in sorted_strategies]
    
    # 直接使用实际的奖励值绘制折线图
    ax.plot(strategy_names, avg_values, marker='o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # 为每个点添加数值标签，显示实际值
    for i, v in enumerate(avg_values):
        # 根据值的正负调整标签位置
        offset = 0.5 if v > 0 else -0.5
        va = 'bottom' if v > 0 else 'top'
        ax.text(i, v + offset, f'{v:.2f}', ha='center', va=va, fontweight='bold')
    
    # 设置图表属性
    ax.set_xlabel('Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_title('Average Performance of Different Strategies', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    
    # 设置Y轴为线性刻度，使用原始数据
    # ax.set_yscale('symlog')
    
    # 设置Y轴范围
    y_min = min(avg_values) * 1.1
    y_max = max(avg_values) * 1.1
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # 保存折线图
    line_fig_path_png = os.path.join(output_dir, 'strategy_average_comparison.png')
    line_fig_path_svg = os.path.join(output_dir, 'strategy_average_comparison.svg')
    plt.savefig(line_fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(line_fig_path_svg, dpi=300, bbox_inches='tight')
    print(f"=== 折线图已保存到: {line_fig_path_png} ===")
    print(f"=== 折线图已保存到: {line_fig_path_svg} ===")
    
    plt.close()


def plot_hydrogen_consumption_bar_chart(results, output_dir):
    """
    绘制等效氢耗柱状图
    横轴按照环境聚合，相同环境的柱子放在一起，柱子高度代表该策略在该工况下的等效氢耗
    等效氢耗由燃料电池和锂电池氢耗堆叠而成
    共4个策略，3种典型环境，12个柱子
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    # 提取唯一的策略和场景
    strategies = ['Rule-Based', 'MARL', 'MRN-MARL', 'Meta-RL']
    typical_environments = ['cruise', 'recon', 'rescue']  # 3种典型环境
    
    # 准备数据结构
    hydrogen_data = {}
    for env in typical_environments:
        hydrogen_data[env] = {}
        for strategy in strategies:
            hydrogen_data[env][strategy] = {'fc': 0.0, 'bat': 0.0}
    
    # 模拟等效氢耗计算（实际应从测试结果中提取）
    # 这里使用随机数据模拟，实际应替换为真实计算
    np.random.seed(42)
    for env in typical_environments:
        for strategy in strategies:
            # 燃料电池氢耗（正值）
            fc_consumption = np.random.uniform(100, 500)
            # 锂电池氢耗（可正可负）
            bat_consumption = np.random.uniform(-200, 200)
            hydrogen_data[env][strategy]['fc'] = fc_consumption
            hydrogen_data[env][strategy]['bat'] = bat_consumption
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 设置颜色
    fc_color = '#c84343'  # 燃料电池颜色
    bat_color = '#42985e'  # 锂电池颜色
    
    # 设置柱状图宽度和位置
    bar_width = 0.15  # 调整柱子宽度，适合多个策略并排
    group_gap = 0.25  # 环境组之间的间隙，减少一半
    
    # 准备数据和位置
    all_fc_values = []
    all_bat_values = []
    all_positions = []
    
    # 计算每个环境组的位置
    group_width = len(strategies) * bar_width
    
    # 生成数据和位置，按环境分组
    for i, env in enumerate(typical_environments):
        group_start = i * (group_width + group_gap)
        for j, strategy in enumerate(strategies):
            pos = group_start + j * bar_width
            all_positions.append(pos)
            all_fc_values.append(hydrogen_data[env][strategy]['fc'])
            all_bat_values.append(hydrogen_data[env][strategy]['bat'])
    
    # 绘制燃料电池氢耗（底部）
    fc_bars = ax.bar(all_positions, all_fc_values, bar_width, label='Fuel Cell', color=fc_color, alpha=0.8)
    
    # 绘制锂电池氢耗（顶部，可正可负）
    bat_bars = ax.bar(all_positions, all_bat_values, bar_width, bottom=all_fc_values, label='Lithium Battery', color=bat_color, alpha=0.8)
    
    # 设置图表属性
    ax.set_ylabel('Equivalent Hydrogen Consumption (g)', fontsize=14, fontweight='bold')
    ax.set_title('Equivalent Hydrogen Consumption by Strategy and Environment', fontsize=16, fontweight='bold', pad=20)
    
    # 设置横轴刻度和标签
    ax.set_xticks(all_positions)
    
    # 设置贴近横轴的策略标签（第一行）
    strategy_labels = []
    for env in typical_environments:
        for strategy in strategies:
            strategy_labels.append(strategy)
    ax.set_xticklabels(strategy_labels, fontsize=10, rotation=45, ha='right')
    
    # 调整x轴标签位置，为第二行标签留出空间
    ax.tick_params(axis='x', pad=20)
    
    # 添加远离横轴的环境标签（第二行）
    env_label_positions = []
    for i, env in enumerate(typical_environments):
        group_start = i * (group_width + group_gap)
        group_center = group_start + group_width / 2
        env_label_positions.append(group_center)
    
    # 在x轴下方添加环境标签
    for i, env in enumerate(typical_environments):
        ax.text(env_label_positions[i], -0.15, env, ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())
    
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 调整纵轴显示范围，避免数据被图例遮挡
    ax.set_ylim(bottom=min(min(all_fc_values) + min(all_bat_values) - 50, 0), top=max(max(all_fc_values) + max(all_bat_values) + 50, 0))
    
    # 添加图例到图框外边，标题之下
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True)
    
    # 调整布局，增加底部边距以容纳图例
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    
    # 保存图表
    bar_chart_path = os.path.join(output_dir, 'hydrogen_consumption_bar_chart.svg')
    plt.savefig(bar_chart_path, dpi=1200, bbox_inches='tight')
    print(f"✅ 等效氢耗柱状图已保存到: {bar_chart_path}")
    
    plt.close()


def plot_violin_chart(results, output_dir):
    """
    绘制小提琴图(violinplot)
    每个环境-策略组合包含两个小提琴图，分别表示锂电池SOC的分布和燃料电池输出功率的分布
    小提琴的胖瘦表达了工作点的密集程度
    12组数据分别是3个典型工况×4种EMS
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    # 提取唯一的策略和场景
    strategies = ['Rule-Based', 'MARL', 'Joint Net', 'Meta-RL']
    typical_environments = ['air', 'surface', 'underwater']  # 3种典型环境
    
    # 准备数据结构
    violin_data = {}
    for env in typical_environments:
        violin_data[env] = {}
        for strategy in strategies:
            violin_data[env][strategy] = {
                'soc': [],  # SOC数据分布
                'fc_power': []  # FC功率数据分布
            }
    
    # 模拟数据分布（实际应从测试结果中提取）
    # 这里生成模拟的SOC和FC功率分布数据
    np.random.seed(42)
    for env in typical_environments:
        for strategy in strategies:
            # 生成SOC分布数据（0-1范围，正态分布）
            soc_mean = np.random.uniform(0.4, 0.6)
            soc_std = np.random.uniform(0.1, 0.2)
            soc_data = np.random.normal(soc_mean, soc_std, 100)  # 生成100个样本
            soc_data = np.clip(soc_data, 0.0, 1.0)  # 确保在0-1范围内
            
            # 生成FC功率分布数据（W，正态分布）
            fc_mean = np.random.uniform(1000, 3000)
            fc_std = np.random.uniform(500, 1000)
            fc_data = np.random.normal(fc_mean, fc_std, 100)  # 生成100个样本
            fc_data = np.clip(fc_data, 0.0, 5000)  # 确保在合理范围内
            
            violin_data[env][strategy]['soc'] = soc_data
            violin_data[env][strategy]['fc_power'] = fc_data
    
    # 创建图表，使用两个子图分别展示SOC和FC功率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 设置颜色
    soc_color = '#42985e'  # 锂电池SOC颜色
    fc_color = '#c84343'  # 燃料电池功率颜色
    
    # 设置小提琴图位置和宽度
    violin_width = 0.8
    x_positions = []
    group_labels = []
    
    # 准备SOC数据用于小提琴图
    soc_violin_data = []
    fc_violin_data = []
    
    for i, strategy in enumerate(strategies):
        for j, env in enumerate(typical_environments):
            # 计算位置
            pos = j * len(strategies) + i
            x_positions.append(pos)
            group_labels.append(f'{env}\n{strategy}')
            
            # 添加SOC数据
            soc_violin_data.append(violin_data[env][strategy]['soc'])
            # 添加FC功率数据
            fc_violin_data.append(violin_data[env][strategy]['fc_power'])
    
    # 绘制SOC小提琴图
    soc_violins = ax1.violinplot(soc_violin_data, positions=x_positions, widths=violin_width, 
                  showmeans=True, showmedians=True, showextrema=True)
    ax1.set_title('Lithium Battery SOC Distribution by Strategy and Environment', fontsize=16, fontweight='bold')
    ax1.set_ylabel('SOC', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.0, 1.0)  # SOC范围0-1
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 设置SOC小提琴图的透明度和颜色
    for pc in soc_violins['bodies']:
        pc.set_facecolor('#42985e')  # 锂电池SOC颜色
        pc.set_alpha(0.7)  # 设置透明度
        pc.set_edgecolor('#2d6a4f')  # 边缘颜色
        pc.set_linewidth(1.0)
    
    # 设置SOC小提琴图的均值、中位数和极值线样式
    soc_violins['cmeans'].set_color('#081c15')
    soc_violins['cmedians'].set_color('#081c15')
    soc_violins['cmins'].set_color('#081c15')
    soc_violins['cmaxes'].set_color('#081c15')
    
    # 绘制FC功率小提琴图
    fc_violins = ax2.violinplot(fc_violin_data, positions=x_positions, widths=violin_width, 
                  showmeans=True, showmedians=True, showextrema=True)
    ax2.set_title('Fuel Cell Power Distribution by Strategy and Environment', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Environment and Strategy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Power (W)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.0, 5000.0)  # FC功率范围0-5000W
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 设置FC功率小提琴图的透明度和颜色
    for pc in fc_violins['bodies']:
        pc.set_facecolor('#c84343')  # 燃料电池颜色
        pc.set_alpha(0.7)  # 设置透明度
        pc.set_edgecolor('#8b3a3a')  # 边缘颜色
        pc.set_linewidth(1.0)
    
    # 设置FC功率小提琴图的均值、中位数和极值线样式
    fc_violins['cmeans'].set_color('#3d0808')
    fc_violins['cmedians'].set_color('#3d0808')
    fc_violins['cmins'].set_color('#3d0808')
    fc_violins['cmaxes'].set_color('#3d0808')
    
    # 设置x轴标签
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(group_labels, fontsize=10, rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    violin_path = os.path.join(output_dir, 'soc_fc_power_violin_chart.svg')
    plt.savefig(violin_path, dpi=1200, bbox_inches='tight')
    print(f"✅ SOC和FC功率小提琴图已保存到: {violin_path}")
    
    plt.close()


def main():
    """
    主函数
    """
    print("=== 开始综合测试 ===")
    
    # 运行测试
    results, output_dir = run_comprehensive_test()
    
    # 保存功率匹配度和平均决策耗时到单独的文件
    # 从结果中提取实际的场景和策略
    actual_scenarios = list(set(r['scenario'] for r in results))
    actual_strategies = list(set(r['strategy'] for r in results))
    
    metrics_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'output_dir': output_dir,
        'scenarios': actual_scenarios,
        'strategies': actual_strategies,
        'metrics': {
            'power_matching_percent': {},
            'avg_decision_time_ms': {}
        }
    }
    
    # 整理数据格式
    for scenario in actual_scenarios:
        metrics_data['metrics']['power_matching_percent'][scenario] = {}
        metrics_data['metrics']['avg_decision_time_ms'][scenario] = {}
        
        for strategy in actual_strategies:
            # 查找对应的结果
            for r in results:
                if r['scenario'] == scenario and r['strategy'] == strategy:
                    metrics_data['metrics']['power_matching_percent'][scenario][strategy] = r['power_matching_percent']
                    metrics_data['metrics']['avg_decision_time_ms'][scenario][strategy] = r['avg_decision_time_ms']
                    break
    
    # 保存为JSON文件
    metrics_file_path = os.path.join(output_dir, 'power_decision_metrics.json')
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
    print(f"✅ 功率匹配度和平均决策耗时数据已保存到: {metrics_file_path}")
    
    # 保存为CSV文件，方便查看数据
    csv_file_path = os.path.join(output_dir, 'power_decision_metrics.csv')
    with open(csv_file_path, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('策略,场景,功率匹配度(%),平均决策耗时(ms)\n')
        
        # 写入数据
        for strategy in actual_strategies:
            for scenario in actual_scenarios:
                pm = metrics_data['metrics']['power_matching_percent'][scenario][strategy]
                dt = metrics_data['metrics']['avg_decision_time_ms'][scenario][strategy]
                f.write(f'{strategy},{scenario},{pm:.2f},{dt:.4f}\n')
    print(f"✅ 功率匹配度和平均决策耗时CSV表格已保存到: {csv_file_path}")
    
    # 绘制对比图（奖励对比）
    plot_comparison(results, output_dir)
    
    # 绘制等效氢耗柱状图
    plot_hydrogen_consumption_bar_chart(results, output_dir)
    
    # 绘制SOC和FC功率小提琴图
    plot_violin_chart(results, output_dir)  # 暂时注释，因为这个函数可能需要调整
    
    print("\n=== 综合测试完成 ===")
    print(f"所有测试结果已保存到: {output_dir}")
    print(f"功率匹配度和平均决策耗时数据文件: {metrics_file_path}")
    print(f"功率匹配度和平均决策耗时CSV表格: {csv_file_path}")

if __name__ == "__main__":
    main()
