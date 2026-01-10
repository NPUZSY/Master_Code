#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快学习/快测试脚本

功能：
1. 加载慢学习训练的权重参数和学习超参数
2. 实现环境分布差异度量（KL散度）
3. 实现更新触发条件
4. 实现在线更新流程
5. 支持在不同环境下进行快测试
6. 支持一键测试超级环境下的所有环境
7. 支持自主指定在线学习超参数

参考文章中的学习流程：
- 环境分布差异度量：KL散度
- 滑动窗口采样-核密度估计
- 双重触发机制：KL散度阈值 + 性能指标阈值
- 在线更新流程：参数备份 → 局部参数优化 → 更新效果验证
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KernelDensity
import json
from datetime import datetime

# 延迟导入matplotlib，仅在需要时导入
def setup_matplotlib():
    """
    设置matplotlib环境
    """
    # 导入matplotlib
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    return matplotlib, plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入项目组件
from Scripts.Chapter5.Meta_RL_Engine import MetaRLPolicy
from Scripts.Chapter3.MARL_Engine import device
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
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

class FastAdaptationTrainer:
    """
    快学习/快测试训练器
    """
    def __init__(self, model_path, hyperparams_path=None, custom_hyperparams=None):
        """
        初始化快学习训练器
        
        Args:
            model_path: 预训练模型路径
            hyperparams_path: 快学习超参数路径
            custom_hyperparams: 自定义超参数
        """
        # 生成唯一的timestamp，用于所有结果保存
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")
        
        # 加载模型
        self.hidden_dim = self._infer_hidden_dim(model_path)
        self.model = MetaRLPolicy(hidden_dim=self.hidden_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"✅ 成功加载模型: {model_path}")
        
        # 加载超参数
        self.hyperparams = self._load_hyperparams(hyperparams_path, custom_hyperparams)
        print(f"✅ 成功加载超参数")
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])
        self.loss_func = nn.SmoothL1Loss()
        
        # 初始化滑动窗口
        self.window_size = self.hyperparams['window_size']
        self.temp_window = []  # 温度滑动窗口
        self.power_window = []  # 功率需求滑动窗口
        
        # 初始化性能指标
        self.performance_metrics = {
            'power_matching': [],
            'hydrogen_growth': [],
            'soc_fluctuation': []
        }
        
        # 初始化训练环境分布（从训练数据中估计）
        self.train_temp_dist = None
        self.train_power_dist = None
        self._estimate_train_distributions()
        
        # 初始化参数备份
        self.params_backup = None
        
    def _infer_hidden_dim(self, model_path):
        """
        从模型文件中推断隐藏层维度
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            hidden_dim: 隐藏层维度
        """
        # 加载模型状态字典
        state_dict = torch.load(model_path, map_location=device)
        
        # 从RNN层参数中推断隐藏层维度
        for key in state_dict.keys():
            if 'rnn.weight_hh_l0' in key:
                # RNN隐藏层维度是weight_hh_l0的形状[out_features, hidden_size]的第二个维度
                return state_dict[key].shape[1]
        
        # 从fc_feature1层参数中推断隐藏层维度
        for key in state_dict.keys():
            if 'fc_feature1.weight' in key:
                return state_dict[key].shape[1]
        
        return 512  # 默认值
    
    def _load_hyperparams(self, hyperparams_path, custom_hyperparams):
        """
        加载超参数
        
        Args:
            hyperparams_path: 超参数文件路径
            custom_hyperparams: 自定义超参数
        
        Returns:
            合并后的超参数字典
        """
        # 默认超参数
        default_hyperparams = {
            "lr": 5e-5,
            "gamma": 0.95,
            "hidden_dim": 512,
            "batch_size": 32,
            "update_steps": 10,
            "kl_threshold": 0.3,
            "window_size": 100,
            "kl_weight_temp": 0.5,
            "kl_weight_power": 0.5,
            "power_matching_threshold": 0.9,
            "hydrogen_growth_threshold": 0.01,
            "soc_fluctuation_threshold": 0.08,
            "performance_check_steps": 50,
            "backup_params": True,
            "optimize_all_params": True,
            "validation_steps": 100,
            "success_reward_iterations": 10,
            "kernel_bandwidth_temp": 2.0,
            "kernel_bandwidth_power": 50.0,
            "density_estimation_method": "gaussian",
            "meta_lr": 5e-6,
            "meta_steps": 5,
            "adaptation_steps": 200,
            "performance_recovery_rate": 0.98
        }
        
        # 从文件加载超参数
        file_hyperparams = {}
        if hyperparams_path and os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r', encoding='utf-8') as f:
                file_hyperparams = json.load(f)
        
        # 合并超参数：默认 → 文件 → 自定义
        hyperparams = default_hyperparams.copy()
        hyperparams.update(file_hyperparams)
        if custom_hyperparams:
            hyperparams.update(custom_hyperparams)
        
        return hyperparams
    
    def _estimate_train_distributions(self):
        """
        估计训练环境的分布
        """
        # 这里使用模拟数据作为训练环境分布
        # 实际应用中，应该从80组训练场景数据中估计
        
        # 生成模拟训练数据
        np.random.seed(42)
        train_temp_data = np.random.normal(25, 5, size=10000)
        train_power_data = np.random.normal(2000, 500, size=10000)
        
        # 使用核密度估计训练环境分布
        self.train_temp_dist = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.hyperparams['kernel_bandwidth_temp']
        )
        self.train_temp_dist.fit(train_temp_data.reshape(-1, 1))
        
        self.train_power_dist = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.hyperparams['kernel_bandwidth_power']
        )
        self.train_power_dist.fit(train_power_data.reshape(-1, 1))
        
        print(f"✅ 成功估计训练环境分布")
    
    def _update_sliding_window(self, temp, power):
        """
        更新滑动窗口
        
        Args:
            temp: 当前温度
            power: 当前功率需求
        """
        # 更新温度滑动窗口
        self.temp_window.append(temp)
        if len(self.temp_window) > self.window_size:
            self.temp_window.pop(0)
        
        # 更新功率需求滑动窗口
        self.power_window.append(power)
        if len(self.power_window) > self.window_size:
            self.power_window.pop(0)
    
    def _estimate_current_distributions(self):
        """
        估计当前环境的分布
        
        Returns:
            temp_dist: 当前温度分布
            power_dist: 当前功率需求分布
        """
        if len(self.temp_window) < self.window_size or len(self.power_window) < self.window_size:
            return None, None
        
        # 估计当前温度分布
        temp_data = np.array(self.temp_window).reshape(-1, 1)
        temp_dist = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.hyperparams['kernel_bandwidth_temp']
        )
        temp_dist.fit(temp_data)
        
        # 估计当前功率需求分布
        power_data = np.array(self.power_window).reshape(-1, 1)
        power_dist = KernelDensity(
            kernel='gaussian', 
            bandwidth=self.hyperparams['kernel_bandwidth_power']
        )
        power_dist.fit(power_data)
        
        return temp_dist, power_dist
    
    def _calculate_kl_divergence(self, p_dist, q_dist, data):
        """
        计算KL散度 D_KL(P||Q)
        
        Args:
            p_dist: 当前分布
            q_dist: 训练分布
            data: 采样数据
        
        Returns:
            kl_divergence: KL散度值
        """
        if p_dist is None or q_dist is None:
            return 0.0
        
        # 计算log P(x) - log Q(x)
        log_p = p_dist.score_samples(data)
        log_q = q_dist.score_samples(data)
        
        # 计算KL散度
        kl_divergence = np.mean(log_p - log_q)
        
        return max(0.0, kl_divergence)  # KL散度非负
    
    def _calculate_total_kl(self):
        """
        计算综合KL散度
        
        Returns:
            total_kl: 综合KL散度值
        """
        # 估计当前分布
        temp_dist, power_dist = self._estimate_current_distributions()
        if temp_dist is None or power_dist is None:
            return 0.0
        
        # 准备数据
        temp_data = np.array(self.temp_window).reshape(-1, 1)
        power_data = np.array(self.power_window).reshape(-1, 1)
        
        # 计算温度KL散度
        kl_temp = self._calculate_kl_divergence(temp_dist, self.train_temp_dist, temp_data)
        
        # 计算功率需求KL散度
        kl_power = self._calculate_kl_divergence(power_dist, self.train_power_dist, power_data)
        
        # 计算综合KL散度
        total_kl = (self.hyperparams['kl_weight_temp'] * kl_temp + 
                   self.hyperparams['kl_weight_power'] * kl_power)
        
        return total_kl
    
    def _update_performance_metrics(self, power_matching, hydrogen_growth, soc_fluctuation):
        """
        更新性能指标
        
        Args:
            power_matching: 功率供需匹配度
            hydrogen_growth: 等效氢耗增长率
            soc_fluctuation: 锂电池SOC波动幅度
        """
        # 更新功率供需匹配度
        self.performance_metrics['power_matching'].append(power_matching)
        if len(self.performance_metrics['power_matching']) > self.hyperparams['performance_check_steps']:
            self.performance_metrics['power_matching'].pop(0)
        
        # 更新等效氢耗增长率
        self.performance_metrics['hydrogen_growth'].append(hydrogen_growth)
        if len(self.performance_metrics['hydrogen_growth']) > self.hyperparams['performance_check_steps']:
            self.performance_metrics['hydrogen_growth'].pop(0)
        
        # 更新锂电池SOC波动幅度
        self.performance_metrics['soc_fluctuation'].append(soc_fluctuation)
        if len(self.performance_metrics['soc_fluctuation']) > self.hyperparams['performance_check_steps']:
            self.performance_metrics['soc_fluctuation'].pop(0)
    
    def _reset_sliding_window(self):
        """
        重置滑动窗口
        """
        self.temp_window = []  # 温度滑动窗口
        self.power_window = []  # 功率需求滑动窗口
    
    def _reset_performance_metrics(self):
        """
        重置性能指标
        """
        self.performance_metrics = {
            'power_matching': [],
            'hydrogen_growth': [],
            'soc_fluctuation': []
        }
    
    def _check_performance_thresholds(self):
        """
        检查性能指标是否超过阈值
        
        Returns:
            performance_anomaly: 是否存在性能异常
        """
        # 检查功率供需匹配度
        if self.performance_metrics['power_matching']:
            avg_power_matching = np.mean(self.performance_metrics['power_matching'])
            if avg_power_matching <= self.hyperparams['power_matching_threshold']:
                return True
        
        # 检查等效氢耗增长率
        if self.performance_metrics['hydrogen_growth']:
            avg_hydrogen_growth = np.mean(self.performance_metrics['hydrogen_growth'])
            if avg_hydrogen_growth >= self.hyperparams['hydrogen_growth_threshold']:
                return True
        
        # 检查锂电池SOC波动幅度
        if self.performance_metrics['soc_fluctuation']:
            avg_soc_fluctuation = np.mean(self.performance_metrics['soc_fluctuation'])
            if avg_soc_fluctuation >= self.hyperparams['soc_fluctuation_threshold']:
                return True
        
        return False
    
    def _should_update(self):
        """
        检查是否应该触发更新
        
        Returns:
            should_update: 是否应该更新
        """
        # 计算综合KL散度
        total_kl = self._calculate_total_kl()
        
        # 检查KL散度阈值
        if total_kl < self.hyperparams['kl_threshold']:
            return False
        
        # 检查性能指标
        performance_anomaly = self._check_performance_thresholds()
        
        return performance_anomaly
    
    def _backup_params(self):
        """
        备份当前参数
        """
        if self.hyperparams['backup_params']:
            self.params_backup = self.model.state_dict().copy()
            print(f"📦 参数备份完成")
    
    def _restore_params(self):
        """
        恢复备份参数
        """
        if self.params_backup is not None:
            self.model.load_state_dict(self.params_backup)
            print(f"🔄 参数恢复完成")
    
    def _optimize_model(self, experiences):
        """
        优化模型参数
        
        Args:
            experiences: 经验数据
        """
        if not experiences:
            return
        
        # 设置模型为训练模式
        self.model.train()
        
        # 优化模型
        for _ in range(self.hyperparams['update_steps']):
            for exp in experiences:
                state = exp['state']
                reward = exp['reward']
                
                # 构建计算图
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
                fc_action_out, bat_action_out, sc_action_out, _ = self.model(state_tensor, None)
                
                # 计算损失
                target = torch.tensor(reward, dtype=torch.float32).to(device)
                loss_fc = self.loss_func(fc_action_out, target.expand_as(fc_action_out)) * 1.5
                loss_bat = self.loss_func(bat_action_out, target.expand_as(bat_action_out))
                loss_sc = self.loss_func(sc_action_out, target.expand_as(sc_action_out))
                
                total_loss = loss_fc + loss_bat + loss_sc
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # 恢复模型为评估模式
        self.model.eval()
        
        print(f"⚡ 模型优化完成，更新步数: {self.hyperparams['update_steps']}")
    
    def _validate_update(self, env, max_steps=100):
        """
        验证更新效果
        
        Args:
            env: 验证环境
            max_steps: 验证步数
        
        Returns:
            update_success: 更新是否成功
        """
        state = env.reset()
        total_reward = 0.0
        success_count = 0
        
        for step in range(max_steps):
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
            fc_action_out, bat_action_out, sc_action_out, _ = self.model(state_tensor, None)
            
            # 贪婪选择动作
            fc_action = torch.argmax(fc_action_out, dim=1).item()
            bat_action = torch.argmax(bat_action_out, dim=1).item()
            sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            total_reward += reward
            state = next_state
            
            # 检查是否成功
            if reward > -100.0:  # 假设奖励大于-100为成功
                success_count += 1
            
            if done:
                break
        
        # 检查是否满足成功条件
        success_rate = success_count / max_steps
        update_success = success_rate >= self.hyperparams['performance_recovery_rate']
        
        print(f"✅ 更新验证完成，成功率: {success_rate:.2f}")
        
        return update_success
    
    def test_single_scenario(self, scenario, max_steps=1000, save_results=True, episodes=1):
        """
        测试单个场景
        
        Args:
            scenario: 场景名称
            max_steps: 最大测试步数
            save_results: 是否保存结果
            episodes: 测试回合数
        
        Returns:
            test_results: 测试结果
        """
        print(f"\n=== 测试场景: {scenario}, 回合数: {episodes} ===")
        
        # 记录测试开始时间
        test_start_time = time.time()
        
        # 初始化总奖励和总步数
        total_reward = 0.0
        total_steps = 0
        
        # 保存每个回合的结果
        all_episode_results = []
        
        for episode in range(episodes):
            print(f"\n--- 回合 {episode+1}/{episodes} ---")
            
            # 创建环境
            env = EnvUltra(scenario_type=scenario)
            state = env.reset()
            
            # 初始化数据收集
            episode_results = {
                'scenario': scenario,
                'episode': episode+1,
                'steps': [],
                'rewards': [],
                'power_fc': [],
                'power_bat': [],
                'power_sc': [],
                'load_demand': [],
                'temperature': [],
                'soc_bat': [],
                'soc_sc': [],
                'kl_values': [],
                'step_times': [],
                'decision_times': [],
                'update_duration': [],
                'updates_triggered': 0
            }
        
            # 初始化回合相关变量
            episode_total_reward = 0.0
            episode_update_triggered = False
            episode_experiences = []
            episode_update_count = 0
            
            # 重置滑动窗口和性能指标
            self._reset_sliding_window()
            self._reset_performance_metrics()
            
            # 回合测试循环
            for step in range(max_steps):
                # 记录步骤开始时间
                step_start_time = time.time()
                
                # 选择动作 - 记录决策开始时间
                decision_start_time = time.time()
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
                fc_action_out, bat_action_out, sc_action_out, _ = self.model(state_tensor, None)
                
                # 贪婪选择动作
                fc_action = torch.argmax(fc_action_out, dim=1).item()
                bat_action = torch.argmax(bat_action_out, dim=1).item()
                sc_action = torch.argmax(sc_action_out, dim=1).item()
                
                action_list = [fc_action, bat_action, sc_action]
                
                # 记录决策结束时间并计算决策耗时
                decision_end_time = time.time()
                decision_duration = decision_end_time - decision_start_time
                episode_results['decision_times'].append(decision_duration)
                
                # 执行动作
                next_state, reward, done, info = env.step(action_list)
                
                # 更新滑动窗口
                temp = info['T_amb']
                power_demand = info['P_load']
                self._update_sliding_window(temp, power_demand)
                
                # 计算功率供需匹配度
                total_supply = info['P_fc'] + info['P_bat'] + info['P_sc']
                power_matching = min(1.0, total_supply / power_demand) if power_demand > 0 else 1.0
                
                # 计算SOC波动幅度（简化计算）
                soc_fluctuation = abs(next_state[5] - state[5])
                
                # 更新性能指标
                self._update_performance_metrics(power_matching, 0.0, soc_fluctuation)  # 简化计算
                
                # 计算综合KL散度
                total_kl = self._calculate_total_kl()
                
                # 收集经验数据
                episode_experiences.append({
                    'state': state,
                    'reward': reward
                })
                
                # 限制经验数据长度
                if len(episode_experiences) > self.hyperparams['batch_size']:
                    episode_experiences.pop(0)
                
                # 检查是否应该触发更新
                if self._should_update() and not episode_update_triggered:
                    print(f"🚀 触发更新，KL散度: {total_kl:.4f}, 步数: {step}")
                    
                    # 记录更新开始时间
                    update_start_time = time.time()
                    
                    # 备份参数
                    self._backup_params()
                    
                    # 优化模型
                    self._optimize_model(episode_experiences)
                    
                    # 验证更新效果
                    if not self._validate_update(env, self.hyperparams['validation_steps']):
                        # 恢复备份参数
                        self._restore_params()
                    else:
                        episode_update_count += 1
                    
                    # 记录更新结束时间
                    update_end_time = time.time()
                    update_duration = update_end_time - update_start_time
                    episode_results['update_duration'].append(update_duration)
                    print(f"⏱️  更新耗时: {update_duration:.4f}秒")
                    
                    episode_update_triggered = True
                
                # 收集测试数据
                episode_results['steps'].append(step)
                episode_results['rewards'].append(reward)
                episode_results['power_fc'].append(info['P_fc'])
                episode_results['power_bat'].append(info['P_bat'])
                episode_results['power_sc'].append(info['P_sc'])
                episode_results['load_demand'].append(power_demand)
                episode_results['temperature'].append(temp)
                episode_results['soc_bat'].append(next_state[5])
                episode_results['soc_sc'].append(next_state[6])
                episode_results['kl_values'].append(total_kl)
                
                # 记录步骤结束时间
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                episode_results['step_times'].append(step_duration)
                
                episode_total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 计算回合统计指标
            episode_avg_reward = episode_total_reward / (step + 1) if step > 0 else 0.0
            episode_results['avg_reward'] = episode_avg_reward
            episode_results['total_reward'] = episode_total_reward
            episode_results['total_steps'] = step + 1
            episode_results['updates_triggered'] = episode_update_count
            
            # 添加到所有回合结果列表
            all_episode_results.append(episode_results)
            
            # 更新总奖励和总步数
            total_reward += episode_total_reward
            total_steps += step + 1
            
            print(f"✅ 回合 {episode+1} 完成")
            print(f"   回合奖励: {episode_total_reward:.2f}")
            print(f"   回合平均奖励: {episode_avg_reward:.4f}")
            print(f"   回合触发更新次数: {episode_update_count}")
        
        # 记录测试结束时间
        test_end_time = time.time()
        total_test_duration = test_end_time - test_start_time
        
        # 计算所有回合的统计指标
        overall_avg_reward = total_reward / total_steps if total_steps > 0 else 0.0
        
        # 计算耗时统计指标
        timing_stats = {
            'total_test_duration': total_test_duration,  # 总测试耗时（秒）
            'avg_episode_duration': 0.0,  # 平均每回合耗时（秒）
            'avg_step_duration_ms': 0.0,  # 平均每步耗时（毫秒）
            'max_step_duration_ms': 0.0,  # 最大单步耗时（毫秒）
            'min_step_duration_ms': float('inf'),  # 最小单步耗时（毫秒）
            'avg_decision_duration_ms': 0.0,  # 平均决策耗时（毫秒）
            'max_decision_duration_ms': 0.0,  # 最大决策耗时（毫秒）
            'min_decision_duration_ms': float('inf'),  # 最小决策耗时（毫秒）
            'total_update_duration_ms': 0.0,  # 总更新耗时（毫秒）
            'avg_update_duration_ms': 0.0,  # 平均更新耗时（毫秒）
            'update_count': 0  # 更新总次数
        }
        
        # 计算各回合的耗时统计
        total_episode_durations = 0.0
        all_step_times = []
        all_decision_times = []
        all_update_times = []
        
        for episode_result in all_episode_results:
            # 计算回合耗时
            if episode_result['step_times']:
                episode_duration = sum(episode_result['step_times'])
                total_episode_durations += episode_duration
            
            # 收集所有步耗时
            all_step_times.extend(episode_result['step_times'])
            
            # 收集所有决策耗时
            if 'decision_times' in episode_result:
                all_decision_times.extend(episode_result['decision_times'])
            
            # 收集所有更新耗时
            all_update_times.extend(episode_result['update_duration'])
        
        # 更新统计指标
        if episodes > 0:
            timing_stats['avg_episode_duration'] = total_episode_durations / episodes
        
        if all_step_times:
            avg_step_s = sum(all_step_times) / len(all_step_times)
            timing_stats['avg_step_duration_ms'] = avg_step_s * 1000
            timing_stats['max_step_duration_ms'] = max(all_step_times) * 1000
            timing_stats['min_step_duration_ms'] = min(all_step_times) * 1000
        
        if all_decision_times:
            avg_decision_s = sum(all_decision_times) / len(all_decision_times)
            timing_stats['avg_decision_duration_ms'] = avg_decision_s * 1000
            timing_stats['max_decision_duration_ms'] = max(all_decision_times) * 1000
            timing_stats['min_decision_duration_ms'] = min(all_decision_times) * 1000
        
        if all_update_times:
            total_update_s = sum(all_update_times)
            avg_update_s = total_update_s / len(all_update_times)
            timing_stats['total_update_duration_ms'] = total_update_s * 1000
            timing_stats['avg_update_duration_ms'] = avg_update_s * 1000
            timing_stats['update_count'] = len(all_update_times)
        
        # 生成最终结果（使用第一个回合的数据作为基础，添加总统计和耗时统计）
        final_results = all_episode_results[0].copy()
        final_results['all_episodes'] = all_episode_results
        final_results['total_reward'] = total_reward
        final_results['total_steps'] = total_steps
        final_results['avg_reward'] = overall_avg_reward
        final_results['episodes'] = episodes
        final_results['timing_stats'] = timing_stats
        
        print(f"\n✅ 场景 {scenario} 测试完成")
        print(f"   总奖励: {total_reward:.2f}")
        print(f"   平均奖励: {overall_avg_reward:.4f}")
        print(f"   总步数: {total_steps}")
        
        # 打印耗时统计
        print(f"\n   ⏱️  耗时统计:")
        print(f"   平均每回合耗时: {timing_stats['avg_episode_duration']:.4f}秒")
        print(f"   平均每步耗时: {timing_stats['avg_step_duration_ms']:.4f}毫秒")
        print(f"   平均决策耗时: {timing_stats['avg_decision_duration_ms']:.4f}毫秒")
        print(f"   最小决策耗时: {timing_stats['min_decision_duration_ms']:.4f}毫秒")
        print(f"   最大决策耗时: {timing_stats['max_decision_duration_ms']:.4f}毫秒")
        print(f"   平均更新耗时: {timing_stats['avg_update_duration_ms']:.4f}毫秒")
        print(f"   总更新次数: {timing_stats['update_count']}")
        
        # 保存测试结果
        if save_results:
            self._save_test_results(final_results)
        
        return final_results
    
    @classmethod
    def plot_power_profiles(cls, all_results, save_path, show_plot=False):
        """
        绘制3种场景的功率分配结果，3行1列子图，参考超级环境plot_scenario_profiles的绘制方式
        
        Args:
            all_results: 所有场景的测试结果
            save_path: 图像保存路径
            show_plot: 是否显示图像
        """
        # 延迟导入matplotlib
        matplotlib, plt = setup_matplotlib()
        
        # 3种场景的顺序和配置
        scenarios = [
            ('cruise', 'Long-Endurance Cruise', '#1f77b4'),
            ('recon', 'Cross-Domain Reconnaissance', '#ff7f0e'),
            ('rescue', 'Emergency Rescue', '#2ca02c')
        ]
        
        # 颜色配置
        power_colors = {
            'load': '#f09639',  # 功率需求
            'fc': '#c84343',     # 燃料电池
            'bat': '#42985e',    # 电池
            'sc': '#8a7ab5'      # 超级电容
        }
        
        # 基础功率和温度参考值
        # 全局字体大小设置
        GLOBAL_FONTSIZE = 16
        
        P_AIR_BASE = 2500
        P_SURFACE_BASE = 1000
        P_UNDERWATER_BASE = 3000
        T_AIR = 0
        T_SURFACE = 20
        T_UNDERWATER = 5
        
        # 创建3行1列子图，共享X轴
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # 绘制每个场景
        for idx, (scenario_type, scenario_label, scenario_color) in enumerate(scenarios):
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # 共享X轴的温度轴
            
            # 获取当前场景的结果
            if scenario_type in all_results:
                scenario_result = all_results[scenario_type]
                
                # 准备数据
                times = scenario_result['steps']
                load_demand = scenario_result['load_demand']
                power_fc = scenario_result['power_fc']
                power_bat = scenario_result['power_bat']
                power_sc = scenario_result['power_sc']
                temperature = scenario_result['temperature']
                soc_bat = scenario_result['soc_bat']
                soc_sc = scenario_result['soc_sc']
                
                # 构建模态阶段信息（简化版，根据时间区间划分）
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
                
                # 绘制功率曲线
                ax1.plot(times, load_demand, label='Power Demand', color=scenario_color, linewidth=1.2, linestyle='-')
                ax1.plot(times, power_fc, label='Fuel Cell', color=power_colors['fc'], linewidth=1.2, linestyle='-')
                ax1.plot(times, power_bat, label='Battery', color=power_colors['bat'], linewidth=1.2, linestyle='-')
                ax1.plot(times, power_sc, label='Super Capacitor', color=power_colors['sc'], linewidth=1.2, linestyle='-')
                
                # 填充功率区域（与超级环境一致，使用场景颜色）
                ax1.fill_between(times, 0, load_demand, color=scenario_color, alpha=0.1)
                
                # 添加基础功率参考线（与超级环境一致）
                ax1.axhline(y=P_AIR_BASE, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Air Base Power ({P_AIR_BASE}W)')
                ax1.axhline(y=P_SURFACE_BASE, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Surface Base Power ({P_SURFACE_BASE}W)')
                ax1.axhline(y=P_UNDERWATER_BASE, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Underwater Base Power ({P_UNDERWATER_BASE}W)')
                
                # 绘制温度曲线（与超级环境一致）
                ax2.plot(times, temperature, color='darkred', linestyle='--', linewidth=1.2, label='Temperature')
                
                # 绘制SOC曲线（保持原有功能）
                ax2.plot(times, [soc * 100 for soc in soc_bat], color='purple', linestyle='-.', linewidth=1.2, label='Battery SOC')
                ax2.plot(times, [soc * 100 for soc in soc_sc], color='cyan', linestyle=':', linewidth=1.2, label='SuperCap SOC')
                
                # 添加温度参考线
                ax2.axhline(y=T_AIR, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Air Temp ({T_AIR}℃)')
                ax2.axhline(y=T_SURFACE, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Surface Temp ({T_SURFACE}℃)')
                ax2.axhline(y=T_UNDERWATER, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Underwater Temp ({T_UNDERWATER}℃)')
                
                # 标注模态阶段（先绘制背景色，再添加标签，确保在最上层）
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
                
                # 添加模态标签（仅标注主要模态，与超级环境一致）
                for mode in modes:
                    if 'switch' not in mode['type']:
                        mid_time = (mode['start'] + mode['end']) / 2
                        ax1.text(mid_time, ax1.get_ylim()[1]*0.7, mode['label'], 
                                ha='center', va='center', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # 设置子图属性
                ax1.set_title(scenario_label, fontsize=14, fontweight='bold', pad=10)  # 与超级环境一致
                ax1.set_ylabel('Power (W)', fontsize=GLOBAL_FONTSIZE, fontweight='bold')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.set_ylim(0, max(max(load_demand), max(power_fc), max(power_bat), max(power_sc)) * 1.1)
                ax1.tick_params(axis='y', labelsize=GLOBAL_FONTSIZE)
                
                ax2.set_ylabel('Temperature (℃) / SOC (%)', fontsize=GLOBAL_FONTSIZE, fontweight='bold', color='darkred')
                ax2.set_ylim(-5, 105)
                ax2.tick_params(axis='y', labelsize=GLOBAL_FONTSIZE, colors='darkred')
                
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
                ax1.set_ylabel('Power (W)', fontsize=GLOBAL_FONTSIZE, fontweight='bold')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)
        
        # 设置X轴
        axes[-1].set_xlabel('Time (s)', fontsize=GLOBAL_FONTSIZE, fontweight='bold')
        axes[-1].set_xlim(0, 1800)  # 设置为1800s
        axes[-1].set_xticks(np.arange(0, 1801, 200))  # 每200s一个刻度
        axes[-1].tick_params(axis='x', labelsize=GLOBAL_FONTSIZE)
        
        # 创建figure级别的共享图例（位于所有Axes之上，与超级环境一致）
        if 'fig_legend_handles' in locals() and 'fig_legend_labels' in locals():
            fig.legend(fig_legend_handles, fig_legend_labels, loc='upper center', fontsize=9, framealpha=0.9, 
                      bbox_to_anchor=(0.5, 0.92), ncol=6)  # 顶部居中，6列布局，与超级环境一致
        
        # 调整布局（与超级环境一致）
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # 调整顶部边距以容纳图例
        
        # 保存图片
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"✅ 功率分配结果图已保存到: {save_path}")
        
        # 显示图像（可选）
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def test_all_scenarios(self, max_steps=1000, save_results=True, show_plot=False, episodes=1):
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
            plot_path = os.path.join(results_dir, "power_distribution_3_scenarios.svg")
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
        
        # 保存单个场景结果
        scenario = results['scenario']
        result_path = os.path.join(results_dir, f"fast_adaptation_result_{scenario}.json")
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
        
        # 保存汇总结果
        summary_path = os.path.join(results_dir, "fast_adaptation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, cls=NumpyEncoder, indent=4)
        
        # 保存超参数
        hyperparams_path = os.path.join(results_dir, "fast_adaptation_hyperparams.json")
        with open(hyperparams_path, 'w', encoding='utf-8') as f:
            json.dump(self.hyperparams, f, cls=NumpyEncoder, indent=4)
        
        print(f"📊 汇总结果已保存到: {summary_path}")
        print(f"📋 超参数已保存到: {hyperparams_path}")
    
    def save_model(self, save_path):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 模型已保存到: {save_path}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='快学习/快测试脚本')
    
    # 核心参数
    parser.add_argument('--model-path', type=str, required=False,
                        help='预训练模型路径（--plot-only模式下可选）')
    parser.add_argument('--hyperparams-path', type=str, default=None,
                        help='快学习超参数路径')
    
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
    
    # 自定义超参数
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--kl-threshold', type=float, default=None,
                        help='KL散度阈值')
    parser.add_argument('--window-size', type=int, default=None,
                        help='滑动窗口大小')
    
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
            plot_path = os.path.join(os.path.dirname(args.plot_only), "power_distribution_fast.svg")
            
            # 创建一个最小化的trainer实例，仅用于调用plot_power_profiles
            if args.model_path:
                trainer = FastAdaptationTrainer(
                    model_path=args.model_path,
                    hyperparams_path=args.hyperparams_path,
                    custom_hyperparams={}
                )
            else:
                # 如果没有提供模型路径，可以直接使用静态方法
                trainer = type('DummyTrainer', (), {
                    'timestamp': datetime.now().strftime("%m%d_%H%M%S"),
                    'plot_power_profiles': FastAdaptationTrainer.plot_power_profiles
                })()
            
            # 调用绘图函数
            trainer.plot_power_profiles(results, plot_path, show_plot=args.show_plot)
            print(f"\n=== 快速绘图完成 ===")
            return
        else:
            print(f"❌ 错误：结果文件{args.plot_only}不存在")
            sys.exit(1)
    
    # 正常测试模式
    if not args.model_path:
        print(f"❌ 错误：正常测试模式下必须提供--model-path参数")
        sys.exit(1)
    
    # 构建自定义超参数
    custom_hyperparams = {}
    if args.lr:
        custom_hyperparams['lr'] = args.lr
    if args.kl_threshold:
        custom_hyperparams['kl_threshold'] = args.kl_threshold
    if args.window_size:
        custom_hyperparams['window_size'] = args.window_size
    
    # 初始化快学习训练器
    trainer = FastAdaptationTrainer(
        model_path=args.model_path,
        hyperparams_path=args.hyperparams_path,
        custom_hyperparams=custom_hyperparams
    )
    
    # 测试场景
    if args.scenario:
        if args.scenario == "classical":
            # 测试经典场景（cruise, recon, rescue）
            trainer.test_all_scenarios(
                max_steps=args.max_steps,
                save_results=args.save_results,
                show_plot=args.show_plot,
                episodes=args.episodes
            )
        else:
            # 测试单个场景
            results = trainer.test_single_scenario(
                scenario=args.scenario,
                max_steps=args.max_steps,
                save_results=args.save_results,
                episodes=args.episodes
            )
            
            # 如果保存结果，绘制单个场景的功率分配图像
            if args.save_results:
                # 绘制单个场景的功率分配图像
                results_dir = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nets/Chap5/fast_adaptation')),
                    trainer.timestamp
                )
                plot_path = os.path.join(results_dir, f"power_distribution_{args.scenario}.svg")
                
                # 创建单个场景的结果字典
                single_result = {args.scenario: results}
                
                # 调用绘图函数
                trainer.plot_power_profiles(single_result, plot_path, show_plot=args.show_plot)
    else:
        # 测试所有场景
        trainer.test_all_scenarios(
            max_steps=args.max_steps,
            save_results=args.save_results,
            show_plot=args.show_plot,
            episodes=args.episodes
        )
    
    print(f"\n=== 快学习/快测试完成 ===")

if __name__ == '__main__':
    main()
