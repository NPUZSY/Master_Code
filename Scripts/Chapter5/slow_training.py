import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim

# 导入公共组件
from Scripts.Chapter5.Meta_RL_Engine import (
    MetaRLEnvironment,
    MetaRLPolicy,
    ResultSaver,
    create_output_dir,
    get_project_root
)
from Scripts.Chapter3.MARL_Engine import device
from Scripts.Chapter5.Env_Ultra import EnvUltra

# ----------------------------------------------------
# 慢训练算法类
# ----------------------------------------------------
class SlowTrainer:
    """
    慢训练算法类，专注于在多种模态上进行扎实的慢训练
    """
    def __init__(self, policy, lr=5e-5, gamma=0.99, hidden_dim=256, num_workers=9):
        self.policy = policy
        # 使用Adam优化器，带有权重衰减
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.gamma = gamma
        # 使用Huber损失，对异常值更鲁棒
        self.loss_func = nn.SmoothL1Loss()
        
        # 9种场景的任务集合
        self.scenarios = [
            'air', 'surface', 'underwater',  # 3种基础场景
            'air_to_surface', 'surface_to_air',  # 切换场景1-2
            'air_to_underwater', 'underwater_to_air',  # 切换场景3-4
            'surface_to_underwater', 'underwater_to_surface'  # 切换场景5-6
        ]
        
        # 设置线程池和线程锁
        self.num_workers = num_workers
        self.model_lock = threading.Lock()
    
    def generate_experiences(self, scenario, max_steps=1000):
        """
        在单个场景上生成经验数据（状态、奖励），不进行梯度更新
        """
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        state = env.reset()
        
        total_reward = 0.0
        steps = 0
        
        # 收集经验数据（仅状态和奖励）
        experiences = []
        
        while steps < max_steps:
            # 每次迭代重新初始化隐藏状态
            hidden = None
            
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
            fc_action_out, bat_action_out, sc_action_out, _ = self.policy(state_tensor, hidden)
            
            # 贪婪选择动作
            fc_action = torch.argmax(fc_action_out, dim=1).item()
            bat_action = torch.argmax(bat_action_out, dim=1).item()
            sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            # 计算目标值，添加燃料电池跟踪负载的奖励项
            P_load = info['P_load']
            P_fc = info['P_fc']
            tracking_reward = -abs(P_load - P_fc) * 0.01  # 鼓励FC接近负载
            
            # 组合奖励
            adjusted_reward = reward + tracking_reward
            
            # 保存经验数据（仅原始数据，不保存计算图相关内容）
            experiences.append({
                'state': state,
                'reward': adjusted_reward
            })
            
            total_reward += adjusted_reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return total_reward / steps if steps > 0 else 0.0, experiences
    
    def update_from_experiences(self, all_experiences):
        """
        从收集的所有经验数据中更新模型：在主线程中重新计算动作并构建计算图
        """
        if not all_experiences:
            return
        
        # 收集所有损失
        all_losses = []
        
        # 在主线程中重新计算所有动作并构建计算图
        for experiences in all_experiences:
            for exp in experiences:
                # 重新计算动作，构建计算图
                state = exp['state']
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
                hidden = None  # 重新初始化隐藏状态
                fc_action_out, bat_action_out, sc_action_out, _ = self.policy(state_tensor, hidden)
                
                # 创建目标张量
                target = torch.tensor(exp['reward'], dtype=torch.float32).to(device)
                
                # 计算损失，增加FC动作的权重，鼓励FC跟踪负载
                loss_fc = self.loss_func(fc_action_out, target.expand_as(fc_action_out)) * 1.5
                loss_bat = self.loss_func(bat_action_out, target.expand_as(bat_action_out))
                loss_sc = self.loss_func(sc_action_out, target.expand_as(sc_action_out))
                
                total_loss = loss_fc + loss_bat + loss_sc
                all_losses.append(total_loss)
        
        # 计算平均损失并更新策略
        if all_losses:
            avg_loss = torch.mean(torch.stack(all_losses))
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()
    
    def train(self, num_epochs=1000, eval_interval=100, save_interval=100, result_saver=None, output_dir=None):
        """
        慢训练主循环
        """
        training_rewards = []
        best_avg_reward = -float('inf')
        
        # 使用tqdm添加epoch进度条
        for epoch in tqdm(range(num_epochs), desc="慢训练进度", unit="epoch"):
            epoch_rewards = []
            all_experiences = []
            
            # 使用多线程并行生成所有场景的经验数据
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 提交所有场景经验生成任务
                future_to_scenario = {executor.submit(self.generate_experiences, scenario, max_steps=1000): scenario for scenario in self.scenarios}
                
                # 使用tqdm显示场景训练进度
                for future in tqdm(future_to_scenario, desc=f"Epoch {epoch}场景经验生成", unit="scenario", leave=False):
                    try:
                        reward, experiences = future.result()
                        epoch_rewards.append(reward)
                        all_experiences.append(experiences)
                    except Exception as e:
                        print(f"  ❌ 场景经验生成失败: {e}")
                        epoch_rewards.append(0.0)
            
            # 在主线程中统一更新模型
            self.update_from_experiences(all_experiences)
            
            avg_reward = np.mean(epoch_rewards)
            training_rewards.append(avg_reward)
            
            # 每eval_interval次迭代进行一次评估
            if epoch % eval_interval == 0:
                print(f"Epoch {epoch}, Average Reward: {avg_reward:.4f}, Best Avg Reward: {best_avg_reward:.4f}")
                
                # 保存最佳模型
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    print(f"  ✅ 最佳模型更新，平均奖励: {best_avg_reward:.4f}")
                    if result_saver:
                        result_saver.save_model(self.policy, "slow_training_model_best")
            
            # 每save_interval次迭代保存一次模型
            if result_saver and output_dir and epoch % save_interval == 0 and epoch > 0:
                result_saver.save_model(self.policy, f"slow_training_model_epoch_{epoch}")
        
        return training_rewards, best_avg_reward

# ----------------------------------------------------
# 慢训练主函数
# ----------------------------------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='慢训练脚本')
    parser.add_argument('--num-epochs', type=int, default=2000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--output-dir', type=str, default='', help='输出目录')
    parser.add_argument('--eval-interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--num-workers', type=int, default=9, help='训练线程数')
    args = parser.parse_args()
    
    # 创建输出目录
    if not args.output_dir:
        output_dir = create_output_dir("slow_training")
    else:
        output_dir = args.output_dir
    
    # 初始化结果保存器
    result_saver = ResultSaver(output_dir)
    
    # 初始化策略网络并移动到设备上
    policy = MetaRLPolicy(hidden_dim=args.hidden_dim).to(device)
    
    # 初始化慢训练器
    trainer = SlowTrainer(policy, lr=args.lr, gamma=args.gamma, hidden_dim=args.hidden_dim, num_workers=args.num_workers)
    
    print("=== 开始慢训练 ===")
    print(f"训练场景: {trainer.scenarios}")
    print(f"学习率: {args.lr}, 隐藏层维度: {args.hidden_dim}, 训练轮次: {args.num_epochs}")
    
    # 执行慢训练
    start_time = time.time()
    training_rewards, best_avg_reward = trainer.train(
        num_epochs=args.num_epochs,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        result_saver=result_saver,
        output_dir=output_dir
    )
    end_time = time.time()
    
    print(f"\n=== 慢训练完成 ===")
    print(f"最佳平均奖励: {best_avg_reward:.4f}")
    print(f"训练耗时: {end_time - start_time:.2f} 秒")
    
    # 保存最终模型
    result_saver.save_model(policy, "slow_training_model_final")
    
    # 保存训练奖励曲线
    rewards_path = os.path.join(output_dir, "training_rewards.npy")
    np.save(rewards_path, training_rewards)
    print(f"✅ 训练奖励曲线已保存到: {rewards_path}")
    
    # 可视化训练奖励曲线
    try:
        from Scripts.Chapter5.Meta_RL_Engine import setup_matplotlib
        matplotlib, plt = setup_matplotlib()
        
        plt.figure(figsize=(12, 6))
        plt.plot(training_rewards, label='Average Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.title('Slow Training Reward Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存训练曲线
        curve_path_svg = os.path.join(output_dir, "training_reward_curve.svg")
        curve_path_png = os.path.join(output_dir, "training_reward_curve.png")
        plt.savefig(curve_path_svg, bbox_inches='tight', dpi=1200)
        plt.savefig(curve_path_png, dpi=1200, bbox_inches='tight')
        print(f"✅ 训练奖励曲线已保存到:")
        print(f"   SVG: {curve_path_svg}")
        print(f"   PNG: {curve_path_png}")
        plt.close()
    except Exception as e:
        print(f"⚠️  无法生成训练奖励曲线: {e}")
    
    # 保存训练配置
    config = {
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "best_avg_reward": best_avg_reward,
        "training_time": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = result_saver.save_results_json(config, "slow_training_config.json")
    print(f"✅ 训练配置已保存到: {config_path}")
    
    print(f"\n所有结果已保存到: {output_dir}")
    
    # 自动运行测试脚本
    print("\n=== 开始自动测试慢学习结果 ===")
    import subprocess
    import sys
    
    # 获取最佳模型路径
    best_model_path = os.path.join(output_dir, "slow_training_model_best.pth")
    
    if os.path.exists(best_model_path):
        # 构建测试命令
        test_cmd = [
            sys.executable,
            "Scripts/Chapter5/test_slow_training.py",
            "--model-path", best_model_path,
            "--hidden-dim", str(args.hidden_dim)
        ]
        
        print(f"执行测试命令: {' '.join(test_cmd)}")
        
        # 执行测试脚本
        result = subprocess.run(test_cmd, cwd=get_project_root(), capture_output=True, text=True)
        
        # 输出测试结果
        print("\n=== 测试脚本输出 ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== 测试脚本错误 ===")
            print(result.stderr)
        
        print(f"\n=== 自动测试完成 ===")
    else:
        print(f"❌ 未找到最佳模型文件: {best_model_path}")
        print("请手动运行测试脚本:")
        print(f"python Scripts/Chapter5/test_slow_training.py --model-path {best_model_path} --hidden-dim {args.hidden_dim}")

if __name__ == "__main__":
    main()
