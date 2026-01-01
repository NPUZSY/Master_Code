import os
import sys
import time
import argparse
import numpy as np

# 设置环境变量，确保动态链接器能找到正确的libstdc++版本
os.environ['LD_LIBRARY_PATH'] = '/home/nwpu/miniconda3/envs/Py310/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

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
    def __init__(self, policy, lr=5e-5, gamma=0.99, hidden_dim=256):
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
    
    def train_on_scene(self, scenario, max_steps=1000):
        """
        在单个场景上训练，优化燃料电池跟踪负载需求
        """
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        state = env.reset()
        
        total_reward = 0.0
        steps = 0
        
        while steps < max_steps:
            # 每次迭代重新初始化隐藏状态，避免计算图重用
            hidden = None
            
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1)
            fc_action_out, bat_action_out, sc_action_out, _ = self.policy(state_tensor, hidden)
            
            # 贪婪选择动作
            fc_action = torch.argmax(fc_action_out, dim=1).item()
            bat_action = torch.argmax(bat_action_out, dim=1).item()
            sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            # 计算目标值，添加燃料电池跟踪负载的奖励项
            # 基于信息中的功率数据计算额外奖励
            P_load = info['P_load']
            P_fc = info['P_fc']
            tracking_reward = -abs(P_load - P_fc) * 0.01  # 鼓励FC接近负载
            
            # 组合奖励
            adjusted_reward = reward + tracking_reward
            target = torch.tensor(adjusted_reward, dtype=torch.float32)
            
            # 计算损失，增加FC动作的权重，鼓励FC跟踪负载
            loss_fc = self.loss_func(fc_action_out, target.expand_as(fc_action_out)) * 1.5  # 增加FC损失权重
            loss_bat = self.loss_func(bat_action_out, target.expand_as(bat_action_out))
            loss_sc = self.loss_func(sc_action_out, target.expand_as(sc_action_out))
            
            total_loss = loss_fc + loss_bat + loss_sc
            
            # 更新策略
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_reward += adjusted_reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return total_reward / steps if steps > 0 else 0.0
    
    def train(self, num_epochs=1000, eval_interval=100, save_interval=100, result_saver=None, output_dir=None):
        """
        慢训练主循环
        """
        training_rewards = []
        best_avg_reward = -float('inf')
        
        for epoch in range(num_epochs):
            epoch_rewards = []
            
            # 在所有9种场景上训练
            for scenario in self.scenarios:
                reward = self.train_on_scene(scenario, max_steps=1000)
                epoch_rewards.append(reward)
            
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
    trainer = SlowTrainer(policy, lr=args.lr, gamma=args.gamma, hidden_dim=args.hidden_dim)
    
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

if __name__ == "__main__":
    main()
