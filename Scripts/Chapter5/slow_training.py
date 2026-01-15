import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# 示例代码
'''
从零开始
nohup python Scripts/Chapter5/slow_training.py \
--num-epochs 1000 \
--load-model-path /home/siyu/Master_Code/nets/Chap5/slow_training/0105_113601/slow_training_model_best.pth \
> logs/0105/0105_1.log 2>&1 &

从joint_net开始
nohup python Scripts/Chapter5/slow_training.py \
--num-epochs 5 \
--from-joint-net /home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/2 \
--num-epochs 1000 \
> logs/0103/0103_2.log 2>&1 &


'''

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim

# 导入公共组件
from Scripts.Chapter5.Meta_RL_Engine import (
    MetaRLPolicy,
    ResultSaver,
    create_output_dir,
    get_project_root
)
from Scripts.Chapter3.MARL_Engine import device, Net
from Scripts.Chapter4.Joint_Net import MultiTaskRNN, JointNet
from Scripts.Chapter5.Env_Ultra import EnvUltra

# ----------------------------------------------------
# 工具函数：从JointNet加载参数到慢学习网络
# ----------------------------------------------------
def load_params_from_joint_net(joint_net_dir, policy):
    """
    从JointNet模型目录加载参数并迁移到MetaRLPolicy网络
    
    Args:
        joint_net_dir: JointNet模型目录路径
        policy: 要加载参数的MetaRLPolicy网络
    """
    print(f"📌 开始从JointNet加载参数: {joint_net_dir}")
    
    # 1. 加载JointNet的三个智能体模型
    agent_names = ["FC", "BAT", "SC"]
    joint_agents = {}
    
    for name in agent_names:
        # 尝试加载模型文件
        model_path = os.path.join(joint_net_dir, f"Joint_Model_{name}.pth")
        if not os.path.exists(model_path):
            # 尝试使用其他文件名格式
            model_path = os.path.join(joint_net_dir, f"slow_training_model_best_{name}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"JointNet模型文件不存在: {model_path}")
        
        # 加载JointNet模型
        try:
            # 创建临时的JointNet结构来加载参数
            temp_rnn = MultiTaskRNN()
            temp_marl = Net(N_STATES=65, N_ACTIONS=32 if name == "FC" else 40 if name == "BAT" else 2)
            temp_joint_net = JointNet(temp_rnn, temp_marl)
            
            temp_joint_net.load_state_dict(torch.load(model_path, map_location=device))
            joint_agents[name] = temp_joint_net
            print(f"✅ 成功加载{name}智能体模型: {model_path}")
        except Exception as e:
            print(f"❌ 加载{name}智能体模型失败: {e}")
            raise
    
    # 2. 获取慢学习网络的当前参数
    slow_state_dict = policy.state_dict()
    
    # 3. 从JointNet模型中提取MARL头部参数并迁移到慢学习网络
    print("\n🔄 开始迁移MARL头部参数...")
    
    # 为每个智能体迁移输出层参数
    for name in agent_names:
        joint_marl_state_dict = joint_agents[name].marl_part.state_dict()
        
        # 映射到慢学习网络的对应输出层
        if name == "FC":
            slow_output_prefix = "fc_fc"
        elif name == "BAT":
            slow_output_prefix = "fc_bat"
        else:  # SC
            slow_output_prefix = "fc_sc"
        
        # 迁移output参数到对应的输出层
        if "output.weight" in joint_marl_state_dict and f"{slow_output_prefix}.weight" in slow_state_dict:
            # 获取JointNet的output层参数
            joint_output_weight = joint_marl_state_dict["output.weight"]  # shape: (action_dim, 64)
            
            # 迁移到慢学习网络的输出层
            # 慢学习网络的输出层输入是32维（fc_feature3的输出）
            # 我们只使用JointNet output层的前32个输入通道
            slow_state_dict[f"{slow_output_prefix}.weight"][:, :32] = joint_output_weight[:, :32]
            
            # 迁移偏置项
            if "output.bias" in joint_marl_state_dict and f"{slow_output_prefix}.bias" in slow_state_dict:
                slow_state_dict[f"{slow_output_prefix}.bias"] = joint_marl_state_dict["output.bias"]
            
            print(f"   ✅ 迁移{name}智能体的output参数到{slow_output_prefix}")
    
    # 只迁移FC智能体的中间层参数到慢学习网络的特征提取层
    print("\n🔄 开始迁移中间层参数...")
    fc_marl_state_dict = joint_agents["FC"].marl_part.state_dict()
    
    # 迁移lay1参数到fc_feature3
    if "lay1.weight" in fc_marl_state_dict and "fc_feature3.weight" in slow_state_dict:
        # 获取JointNet的lay1层参数
        joint_lay1_weight = fc_marl_state_dict["lay1.weight"]  # shape: (64, 64)
        
        # 慢学习网络的fc_feature3输入是64维，输出是32维
        # 我们只使用JointNet lay1层的前32个输出通道和前32个输入通道
        slow_state_dict["fc_feature3.weight"][:, :32] = joint_lay1_weight[:32, :32]
        
        # 迁移偏置项
        if "lay1.bias" in fc_marl_state_dict and "fc_feature3.bias" in slow_state_dict:
            slow_state_dict["fc_feature3.bias"][:32] = fc_marl_state_dict["lay1.bias"][:32]
        
        print(f"   ✅ 迁移FC智能体的lay1参数到fc_feature3")
    
    # 迁移input层参数到fc_feature2
    if "input.weight" in fc_marl_state_dict and "fc_feature2.weight" in slow_state_dict:
        # 获取JointNet的input层参数
        joint_input_weight = fc_marl_state_dict["input.weight"]  # shape: (64, 65)
        
        # 慢学习网络的fc_feature2输入是128维，输出是64维
        # 我们只使用JointNet input层的前64个输出通道和前64个输入通道
        # 注意：JointNet的input层输入是65维（64+1），我们跳过reg_out部分，只使用64维特征
        slow_state_dict["fc_feature2.weight"][:64, :64] = joint_input_weight[:, 1:65]  # 跳过JointNet的reg_out部分
        
        # 迁移偏置项
        if "input.bias" in fc_marl_state_dict and "fc_feature2.bias" in slow_state_dict:
            slow_state_dict["fc_feature2.bias"][:64] = fc_marl_state_dict["input.bias"]
        
        print(f"   ✅ 迁移FC智能体的input参数到fc_feature2")
    
    # 4. 更新慢学习网络的所有参数
    policy.load_state_dict(slow_state_dict)
    
    print("\n✅ 所有JointNet参数迁移完成！")
    return policy

# ----------------------------------------------------
# 慢训练算法类
# ----------------------------------------------------
class SlowTrainer:
    """
    慢训练算法类，专注于在多种模态上进行扎实的慢训练
    使用传统DQN训练逻辑：双网络结构、经验回放、Bellman方程
    """
    def __init__(self, policy, lr=5e-4, gamma=0.99, hidden_dim=256, num_workers=9, epsilon=0.1, pool_size=100):
        self.policy = policy
        # 创建目标网络
        self.target_policy = MetaRLPolicy(hidden_dim=hidden_dim).to(device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()  # 目标网络设置为评估模式
        
        # 使用Adam优化器，带有权重衰减
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        # 添加学习率调度器，当奖励连续100轮不提升时，学习率乘以0.5
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=1000, factor=0.8
        )
        # 跟踪当前学习率，用于日志提示
        self.current_lr = lr
        self.gamma = gamma
        # 探索率参数
        self.epsilon = epsilon
        
        # DQN训练参数
        self.target_replace_iter = 10  # 目标网络更新频率
        self.learn_step_counter = 0  # 学习步数计数器
        self.batch_size = 32  # 批次大小
        self.pool_size = pool_size  # 池大小参数，用于计算经验池容量
        
        # 经验回放池 - 固定大小，不随环境动态调整
        self.memory = []
        # 经验池大小固定为：1800s/场景 * 9个场景 * pool_size参数
        self.memory_capacity = 1000 * 9 * self.pool_size
        # 经验池填满标志
        self.memory_full_notified = False
        
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
        在单个场景上生成完整的经验数据（状态、动作、奖励、下一状态），用于后续训练
        """
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        state = env.reset()
        
        # 经验池大小已固定，无需动态调整
        # 每个场景最大为1800s，共9个场景，乘以pool_size参数
        
        total_reward = 0.0
        steps = 0
        
        # 收集完整的经验数据
        experiences = []
        
        while steps < max_steps:
            # 每次迭代重新初始化隐藏状态
            hidden = None
            
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(1).to(device)
            fc_action_out, bat_action_out, sc_action_out, _ = self.policy(state_tensor, hidden)
            
            # 使用epsilon-greedy策略选择动作
            # 燃料电池智能体
            if np.random.random() < self.epsilon:
                fc_action = np.random.randint(0, fc_action_out.shape[1])
            else:
                fc_action = torch.argmax(fc_action_out, dim=1).item()
            
            # 电池智能体
            if np.random.random() < self.epsilon:
                bat_action = np.random.randint(0, bat_action_out.shape[1])
            else:
                bat_action = torch.argmax(bat_action_out, dim=1).item()
            
            # 超级电容智能体
            if np.random.random() < self.epsilon:
                sc_action = np.random.randint(0, sc_action_out.shape[1])
            else:
                sc_action = torch.argmax(sc_action_out, dim=1).item()
            
            action_list = [fc_action, bat_action, sc_action]
            
            # 执行动作
            next_state, reward, done, info = env.step(action_list)
            
            # 计算目标值，添加燃料电池跟踪负载的奖励项
            P_load = info['P_load']
            P_fc = info['P_fc']
            # tracking_reward = -abs(P_load - P_fc) * 0.01  # 鼓励FC接近负载
            
            # 组合奖励
            # adjusted_reward = reward + tracking_reward
            
            # 保存完整的经验数据
            experiences.append({
                'state': state,
                'action': action_list,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return total_reward, experiences
    
    def update_from_experiences(self, all_experiences):
        """
        从收集的所有经验数据中更新模型：使用传统DQN训练逻辑
        """
        if not all_experiences:
            return
        
        # 1. 将生成的经验数据存储到经验回放池中
        for experiences in all_experiences:
            for exp in experiences:
                # 存储经验到回放池
                self.memory.append(exp)
                # 如果回放池超过容量，删除最旧的经验
                if len(self.memory) > self.memory_capacity:
                    self.memory.pop(0)
                
                # 检查经验池是否填满，并打印通知（只通知一次）
                if len(self.memory) >= self.memory_capacity and not self.memory_full_notified:
                    print(f"[INFO] 经验池已填满！当前容量: {len(self.memory)}/{self.memory_capacity}")
                    print(f"[INFO] 经验池大小配置: 1800s/场景 * 9场景 * pool_size({self.pool_size}) = {1800 * 9 * self.pool_size}")
                    self.memory_full_notified = True
        
        # 2. 当经验池足够大时，进行训练
        if len(self.memory) < self.batch_size:
            return
        
        # 3. 随机采样一批经验
        sample_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch_experiences = [self.memory[i] for i in sample_indices]
        
        # 4. 准备训练数据
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in batch_experiences:
            states.append(exp['state'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            next_states.append(exp['next_state'])
            dones.append(exp['done'])
        
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(device)
        dones_tensor = torch.BoolTensor(np.array(dones)).to(device)
        
        # 5. 使用当前网络计算Q值（q_eval）
        fc_q_eval, bat_q_eval, sc_q_eval, _ = self.policy(states_tensor, None)
        
        # 6. 使用目标网络计算下一状态的最大Q值（q_next）
        with torch.no_grad():
            fc_q_next, bat_q_next, sc_q_next, _ = self.target_policy(next_states_tensor, None)
            fc_q_next_max = fc_q_next.max(dim=1)[0]
            bat_q_next_max = bat_q_next.max(dim=1)[0]
            sc_q_next_max = sc_q_next.max(dim=1)[0]
        
        # 7. 计算目标Q值（q_target = r + gamma * q_next）
        fc_q_target = rewards_tensor + self.gamma * fc_q_next_max * (~dones_tensor)
        bat_q_target = rewards_tensor + self.gamma * bat_q_next_max * (~dones_tensor)
        sc_q_target = rewards_tensor + self.gamma * sc_q_next_max * (~dones_tensor)
        
        # 8. 提取实际动作对应的Q值
        actions = np.array(actions)
        fc_actions = actions[:, 0].tolist()
        bat_actions = actions[:, 1].tolist()
        sc_actions = actions[:, 2].tolist()
        
        # 转换为张量
        fc_actions_tensor = torch.LongTensor(fc_actions).unsqueeze(1).to(device)
        bat_actions_tensor = torch.LongTensor(bat_actions).unsqueeze(1).to(device)
        sc_actions_tensor = torch.LongTensor(sc_actions).unsqueeze(1).to(device)
        
        # 提取对应动作的Q值
        fc_q_eval_selected = fc_q_eval.gather(1, fc_actions_tensor).squeeze(1)
        bat_q_eval_selected = bat_q_eval.gather(1, bat_actions_tensor).squeeze(1)
        sc_q_eval_selected = sc_q_eval.gather(1, sc_actions_tensor).squeeze(1)
        
        # 9. 计算损失
        loss_func = nn.MSELoss()
        fc_loss = loss_func(fc_q_eval_selected, fc_q_target)
        bat_loss = loss_func(bat_q_eval_selected, bat_q_target)
        sc_loss = loss_func(sc_q_eval_selected, sc_q_target)
        
        # 总损失（三个智能体的损失之和）
        total_loss = fc_loss + bat_loss + sc_loss
        
        # 10. 反向传播更新当前网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 11. 定期更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            print(f"📌 目标网络已更新（步数: {self.learn_step_counter}）")
    
    def train(self, num_epochs=1000, eval_interval=100, save_interval=100, result_saver=None, output_dir=None):
        """
        慢训练主循环
        """
        training_rewards = []
        best_avg_reward = -float('inf')
        
        # 使用tqdm添加epoch进度条
        pbar = tqdm(range(num_epochs), desc="慢训练进度", unit="epoch")
        for epoch in pbar:
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
            
            # 更新学习率调度器
            self.lr_scheduler.step(avg_reward)
            
            # 检查学习率是否变化并输出日志
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 在tqdm进度条上显示当前的奖励值和学习率
            pbar.set_postfix({"当前奖励": f"{avg_reward:.4f}", "当前学习率": f"{new_lr:.6f}"})
            if new_lr != self.current_lr:
                print(f"📉 学习率已更新: {self.current_lr:.6f} → {new_lr:.6f}")
                self.current_lr = new_lr
            
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
    parser.add_argument('--num-epochs', type=int, default=1000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--epsilon', type=float, default=0.1, help='贪心率/探索率')
    parser.add_argument('--output-dir', type=str, default='', help='输出目录')
    parser.add_argument('--eval-interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--num-workers', type=int, default=9, help='训练线程数')
    parser.add_argument('--pool-size', type=int, default=100, help='池大小（用于计算经验池容量）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于确保训练可复现')
    parser.add_argument('--load-model-path', type=str, default='', help='要加载的预训练慢学习模型路径，用于继续训练')
    parser.add_argument('--from-joint-net', type=str, default='', help='要加载的JointNet模型目录，用于从JointNet继续训练')
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建输出目录
    if not args.output_dir:
        output_dir = create_output_dir("slow_training")
    else:
        output_dir = args.output_dir
    
    # 初始化结果保存器
    result_saver = ResultSaver(output_dir)
    
    # 初始化策略网络并移动到设备上
    policy = MetaRLPolicy(hidden_dim=args.hidden_dim).to(device)
    
    # 加载预训练模型（如果提供）
    if args.load_model_path and args.from_joint_net:
        print("❌ 错误：--load-model-path 和 --from-joint-net 不能同时使用")
        raise ValueError("--load-model-path 和 --from-joint-net 不能同时使用")
    elif args.load_model_path:
        # 从慢学习模型加载
        if os.path.exists(args.load_model_path):
            try:
                policy.load_state_dict(torch.load(args.load_model_path, map_location=device))
                print(f"✅ 成功加载预训练慢学习模型: {args.load_model_path}")
            except Exception as e:
                print(f"❌ 加载预训练慢学习模型失败: {e}")
                raise
        else:
            print(f"❌ 预训练模型文件不存在: {args.load_model_path}")
            raise FileNotFoundError(f"预训练模型文件不存在: {args.load_model_path}")
    elif args.from_joint_net:
        # 从JointNet模型加载参数
        if os.path.exists(args.from_joint_net):
            try:
                policy = load_params_from_joint_net(args.from_joint_net, policy)
                print(f"✅ 成功从JointNet加载参数: {args.from_joint_net}")
            except Exception as e:
                print(f"❌ 从JointNet加载参数失败: {e}")
                raise
        else:
            print(f"❌ JointNet模型目录不存在: {args.from_joint_net}")
            raise FileNotFoundError(f"JointNet模型目录不存在: {args.from_joint_net}")
    
    # 初始化慢训练器
    trainer = SlowTrainer(policy, lr=args.lr, gamma=args.gamma, hidden_dim=args.hidden_dim, num_workers=args.num_workers, epsilon=args.epsilon, pool_size=args.pool_size)
    
    print("=== 开始慢训练 ===")
    print(f"训练场景: {trainer.scenarios}")
    print(f"学习率: {args.lr}, 折扣因子: {args.gamma}, 贪心率: {args.epsilon}, 隐藏层维度: {args.hidden_dim}, 训练轮次: {args.num_epochs}")
    print(f"训练线程数: {args.num_workers}, 经验池大小参数: {args.pool_size}")
    
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
        "epsilon": args.epsilon,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "best_avg_reward": best_avg_reward,
        "training_time": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = result_saver.save_results_json(config, "slow_training_config.json")
    print(f"✅ 训练配置已保存到: {config_path}")
    
    # 生成快学习超参数
    fast_learning_hyperparams = {
        # 基础学习超参数
        "lr": args.lr * 0.1,  # 快学习使用较小的学习率
        "gamma": args.gamma,
        "hidden_dim": args.hidden_dim,
        "batch_size": 32,
        "update_steps": 10,  # 每次更新的步数
        
        # KL散度相关参数
        "kl_threshold": 0.3,  # 更新触发KL散度阈值
        "window_size": 100,  # 滑动窗口大小
        "kl_weight_temp": 0.5,  # 温度KL散度权重
        "kl_weight_power": 0.5,  # 功率需求KL散度权重
        
        # 性能指标阈值
        "power_matching_threshold": 0.9,  # 功率供需匹配度阈值
        "hydrogen_growth_threshold": 0.1,  # 等效氢耗增长率阈值
        "soc_fluctuation_threshold": 0.08,  # 锂电池SOC波动幅度阈值
        "performance_check_steps": 50,  # 性能检查步数
        
        # 更新流程参数
        "backup_params": True,  # 是否备份参数
        "optimize_all_params": True,  # 是否优化所有参数
        "validation_steps": 100,  # 验证步数
        "success_reward_iterations": 10,  # 连续成功迭代次数
        
        # 核密度估计参数
        "kernel_bandwidth_temp": 2.0,  # 温度带宽
        "kernel_bandwidth_power": 50.0,  # 功率需求带宽
        "density_estimation_method": "gaussian",  # 核密度估计方法
        
        # 元学习相关参数
        "meta_lr": args.lr * 0.01,  # 元学习率
        "meta_steps": 5,  # 元学习步数
        "adaptation_steps": 200,  # 适配步数
        "performance_recovery_rate": 0.98  # 性能恢复率
    }
    
    # 保存快学习超参数
    fast_hyperparams_path = result_saver.save_results_json(fast_learning_hyperparams, "fast_learning_hyperparams.json")
    print(f"✅ 快学习超参数已保存到: {fast_hyperparams_path}")
    
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
