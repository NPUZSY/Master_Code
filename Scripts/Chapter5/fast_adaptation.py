import os
import sys
import time
import argparse
import numpy as np

# 系统已修复libstdc++.so.6问题，不再需要环境变量设置

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim

# 导入公共组件
from Scripts.Chapter5.Meta_RL_Engine import (
    MetaRLEnvironment,
    MetaRLPolicy,
    FastAdapter,
    ResultSaver,
    create_output_dir,
    load_model
)
from Scripts.Chapter5.Env_Ultra import EnvUltra

# ----------------------------------------------------
# 快适配训练器类
# ----------------------------------------------------
class FastAdaptationTrainer:
    """
    快适配训练器，用于在新任务上快速调整慢训练得到的模型
    """
    def __init__(self, base_policy, adaptation_lr=5e-5, kl_threshold=0.15, adaptation_steps=100):
        """
        初始化快适配训练器
        
        Args:
            base_policy: 慢训练得到的基础模型
            adaptation_lr: 快适配的学习率
            kl_threshold: 触发快适配的KL散度阈值
            adaptation_steps: 快适配的步数
        """
        self.base_policy = base_policy
        self.adaptation_lr = adaptation_lr
        self.kl_threshold = kl_threshold
        self.adaptation_steps = adaptation_steps
        
        # 初始化快速适配器
        self.adapter = FastAdapter(self.base_policy, kl_threshold=self.kl_threshold)
        
        # 快适配使用的优化器
        self.adaptation_optimizer = None
        
        # 使用Huber损失，对异常值更鲁棒
        self.loss_func = nn.SmoothL1Loss()
    
    def adapt_to_new_task(self, task_data, new_scenario):
        """
        快速适配到新任务
        
        Args:
            task_data: 新任务的数据
            new_scenario: 新场景类型
        """
        # 创建新场景的环境
        env = EnvUltra(scenario_type=new_scenario)
        
        # 检查是否需要进行快适配
        current_state = {
            'power': np.random.normal(2000, 500, 100),
            'temperature': np.random.normal(10, 10, 100)
        }
        
        if self.adapter.should_update(current_state, task_data):
            print(f"🔄 检测到环境变化，开始对场景 '{new_scenario}' 进行快适配...")
            
            # 获取适配后的模型
            adapted_policy = self.adapter.adapt(task_data, self.adaptation_steps)
            
            # 在新场景上进行快速微调
            adapted_policy = self._fine_tune_on_scenario(adapted_policy, new_scenario)
            
            print(f"✅ 场景 '{new_scenario}' 快适配完成")
            return adapted_policy
        else:
            print(f"✅ 环境稳定，场景 '{new_scenario}' 无需快适配")
            return self.base_policy
    
    def _fine_tune_on_scenario(self, policy, scenario, max_steps=1000):
        """
        在特定场景上进行快速微调
        """
        # 设置模型为训练模式
        policy.train()
        
        # 初始化优化器
        self.adaptation_optimizer = optim.Adam(policy.parameters(), lr=self.adaptation_lr, weight_decay=1e-5)
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        
        # 进行多次适配迭代
        for adapt_step in range(self.adaptation_steps):
            state = env.reset()
            episode_loss = 0.0
            steps = 0
            
            while steps < max_steps:
                # 每次迭代重新初始化隐藏状态，避免计算图重用
                hidden = None
                
                # 选择动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1)
                fc_action_out, bat_action_out, sc_action_out, _ = policy(state_tensor, hidden)
                
                # 贪婪选择动作
                fc_action = torch.argmax(fc_action_out, dim=1).item()
                bat_action = torch.argmax(bat_action_out, dim=1).item()
                sc_action = torch.argmax(sc_action_out, dim=1).item()
                
                action_list = [fc_action, bat_action, sc_action]
                
                # 执行动作
                next_state, reward, done, info = env.step(action_list)
                
                # 计算目标值
                target = torch.tensor(reward, dtype=torch.float32)
                
                # 计算损失
                loss_fc = self.loss_func(fc_action_out, target.expand_as(fc_action_out))
                loss_bat = self.loss_func(bat_action_out, target.expand_as(bat_action_out))
                loss_sc = self.loss_func(sc_action_out, target.expand_as(sc_action_out))
                
                total_loss = loss_fc + loss_bat + loss_sc
                
                # 更新策略
                self.adaptation_optimizer.zero_grad()
                total_loss.backward()
                self.adaptation_optimizer.step()
                
                episode_loss += total_loss.item()
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # 打印适配进度
            if (adapt_step + 1) % 10 == 0:
                avg_loss = episode_loss / steps if steps > 0 else 0.0
                print(f"  适配进度: {adapt_step + 1}/{self.adaptation_steps}, 平均损失: {avg_loss:.4f}")
        
        # 恢复为评估模式
        policy.eval()
        return policy
    
    def adapt_to_all_scenarios(self, scenarios):
        """
        对所有新场景进行快适配
        
        Args:
            scenarios: 需要适配的场景列表
        """
        adapted_policies = {}
        
        # 创建元环境，用于生成任务数据
        meta_env = MetaRLEnvironment()
        
        for scenario in scenarios:
            # 生成该场景的任务数据
            task_data = meta_env.generate_mode_data(scenario, duration=200)
            
            # 进行快适配
            adapted_policy = self.adapt_to_new_task(task_data, scenario)
            adapted_policies[scenario] = adapted_policy
        
        return adapted_policies
    
    def test_adapted_policy(self, policy, scenario, max_steps=1000):
        """
        测试适配后的策略
        
        Args:
            policy: 适配后的策略
            scenario: 测试场景
            max_steps: 最大测试步数
        
        Returns:
            测试结果，包括功率分配数据和性能指标
        """
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        state = env.reset()
        
        # 初始化数据存储
        power_fc = []
        power_bat = []
        power_sc = []
        load_power = []
        soc_bat = []
        soc_sc = []
        rewards = []
        
        total_reward = 0.0
        steps = 0
        
        while steps < max_steps:
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1)
            fc_action_out, bat_action_out, sc_action_out, _ = policy(state_tensor)
            
            # 贪婪选择动作
            fc_action = torch.argmax(fc_action_out, dim=1).item()
            bat_action = torch.argmax(bat_action_out, dim=1).item()
            sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            # 记录数据
            power_fc.append(float(next_state[2]))
            power_bat.append(float(next_state[3]))
            power_sc.append(float(next_state[4]))
            load_power.append(float(state[0]))  # 负载功率是当前状态的第一个元素
            soc_bat.append(float(next_state[5]))
            soc_sc.append(float(next_state[6]))
            rewards.append(reward)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # 计算性能指标
        avg_reward = np.mean(rewards) if rewards else 0.0
        std_reward = np.std(rewards) if rewards else 0.0
        
        # 准备功率分配数据
        power_data = {
            'power_fc': power_fc,
            'power_bat': power_bat,
            'power_sc': power_sc,
            'load_power': load_power,
            'soc_bat': soc_bat,
            'soc_sc': soc_sc,
            'temperature': [env.temperature] * steps  # 假设环境温度恒定
        }
        
        # 准备性能数据
        performance = {
            'scenario': scenario,
            'total_steps': steps,
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'power_fc_avg': np.mean(power_fc) if power_fc else 0.0,
            'power_bat_avg': np.mean(power_bat) if power_bat else 0.0,
            'power_sc_avg': np.mean(power_sc) if power_sc else 0.0,
            'soc_bat_min': np.min(soc_bat) if soc_bat else 0.0,
            'soc_bat_max': np.max(soc_bat) if soc_bat else 0.0,
            'soc_sc_min': np.min(soc_sc) if soc_sc else 0.0,
            'soc_sc_max': np.max(soc_sc) if soc_sc else 0.0
        }
        
        return power_data, performance

# ----------------------------------------------------
# 快适配主函数
# ----------------------------------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='快适配脚本')
    parser.add_argument('--base-model-path', type=str, required=True, help='慢训练模型文件路径')
    parser.add_argument('--adaptation-lr', type=float, default=5e-5, help='快适配学习率')
    parser.add_argument('--kl-threshold', type=float, default=0.15, help='KL散度阈值')
    parser.add_argument('--adaptation-steps', type=int, default=100, help='快适配步数')
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--output-dir', type=str, default='', help='输出目录')
    
    # 可选参数：指定需要适配的场景
    parser.add_argument('--scenarios', nargs='+', type=str, default=None, 
                        choices=['air', 'surface', 'underwater', 
                                 'air_to_surface', 'surface_to_air',
                                 'air_to_underwater', 'underwater_to_air',
                                 'surface_to_underwater', 'underwater_to_surface'],
                        help='需要适配的场景列表，默认适配所有9种场景')
    
    # 测试参数
    parser.add_argument('--test-steps', type=int, default=1000, help='测试步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not args.output_dir:
        output_dir = create_output_dir("fast_adaptation")
    else:
        output_dir = args.output_dir
    
    # 初始化结果保存器
    result_saver = ResultSaver(output_dir)
    
    # 初始化基础策略网络
    base_policy = MetaRLPolicy(hidden_dim=args.hidden_dim)
    
    # 加载慢训练模型
    if not load_model(base_policy, args.base_model_path):
        print("❌ 无法加载慢训练模型，快适配失败")
        sys.exit(1)
    
    # 设置模型为评估模式
    base_policy.eval()
    
    # 初始化快适配训练器
    trainer = FastAdaptationTrainer(
        base_policy=base_policy,
        adaptation_lr=args.adaptation_lr,
        kl_threshold=args.kl_threshold,
        adaptation_steps=args.adaptation_steps
    )
    
    # 确定需要适配的场景
    if args.scenarios:
        scenarios_to_adapt = args.scenarios
    else:
        # 默认适配所有9种场景
        scenarios_to_adapt = [
            'air', 'surface', 'underwater',
            'air_to_surface', 'surface_to_air',
            'air_to_underwater', 'underwater_to_air',
            'surface_to_underwater', 'underwater_to_surface'
        ]
    
    print("=== 开始快适配 ===")
    print(f"基础模型: {args.base_model_path}")
    print(f"适配场景: {scenarios_to_adapt}")
    print(f"适配学习率: {args.adaptation_lr}")
    print(f"KL阈值: {args.kl_threshold}")
    print(f"适配步数: {args.adaptation_steps}")
    
    # 执行快适配
    start_time = time.time()
    adapted_policies = trainer.adapt_to_all_scenarios(scenarios_to_adapt)
    end_time = time.time()
    
    print(f"\n=== 快适配完成 ===")
    print(f"适配场景数量: {len(adapted_policies)}")
    print(f"适配耗时: {end_time - start_time:.2f} 秒")
    
    # 保存适配后的模型和测试结果
    all_test_results = {
        "adaptation_config": {
            "base_model_path": args.base_model_path,
            "adaptation_lr": args.adaptation_lr,
            "kl_threshold": args.kl_threshold,
            "adaptation_steps": args.adaptation_steps,
            "adapted_scenarios": list(adapted_policies.keys()),
            "adaptation_time": end_time - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_params": {
                "test_steps": args.test_steps,
                "seed": args.seed
            }
        },
        "scenario_results": {}
    }
    
    # 测试每个适配后的模型
    for scenario, policy in adapted_policies.items():
        print(f"\n🔍 测试场景: {scenario}")
        
        # 保存适配后的模型
        if policy is not base_policy:  # 只保存经过适配的模型
            model_name = f"fast_adapted_model_{scenario}"
            result_saver.save_model(policy, model_name)
        
        # 测试模型性能
        power_data, performance = trainer.test_adapted_policy(policy, scenario, max_steps=args.test_steps)
        
        # 保存功率分配图
        plot_filename = f"power_distribution_{scenario}.svg"
        result_saver.save_power_distribution_plot(power_data, scenario, filename=plot_filename)
        
        # 保存性能数据
        all_test_results["scenario_results"][scenario] = {
            "performance": performance,
            "power_data": power_data
        }
        
        print(f"✅ 场景 '{scenario}' 测试完成")
        print(f"   总奖励: {performance['total_reward']:.4f}")
        print(f"   平均奖励: {performance['average_reward']:.4f}")
    
    # 保存基础模型作为参考
    result_saver.save_model(base_policy, "base_slow_trained_model")
    
    # 保存所有测试结果
    result_saver.save_results_json(all_test_results, "fast_adaptation_test_results.json")
    
    print(f"\n=== 所有适配和测试完成 ===")
    print(f"所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
