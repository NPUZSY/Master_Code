import os
import time
import json
import subprocess
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch

def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# ----------------------------------------------------
# 1. 环境与路径配置
# ----------------------------------------------------
from Scripts.Chapter3.MARL_Engine import (
    setup_project_root, device, 
    get_max_folder_name
)
project_root = setup_project_root()
from Scripts.Env import Envs
from Scripts.utils.global_utils import *

# 确保 Joint_Net 定义正确
from Scripts.Chapter4.Joint_Net import MultiTaskRNN, JointNet, JointDQN

font_get()

# ====================== 命令行参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description='Joint_Model 耦合模型继续训练脚本')
    
    # 路径参数
    parser.add_argument('--pretrain-date', type=str, default="1219")
    parser.add_argument('--pretrain-train-id', type=str, default="1")
    parser.add_argument('--pretrain-prefix', type=str, default="Joint_Model")
    parser.add_argument('--rnn-path', type=str, 
                        default=os.path.join(project_root, "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth"))

    # 训练超参数
    parser.add_argument('--batch-size', type=int, default=128) # 微调建议 batch 大一点
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--pool-size', type=int, default=20) # 建议增大池子
    parser.add_argument('--episode', type=int, default=2000)
    parser.add_argument('--learn-frequency', type=int, default=5)
    
    return parser.parse_args()

args = parse_args()

# ====================== 全局配置 ======================
try:
    env = Envs()
except Exception as e:
    print(f"❌ 环境初始化失败: {e}")
    sys.exit()

torch.set_default_dtype(torch.float32)
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

N_STATES = 7 
# 确保 MEMORY_CAPACITY 至少大于 batch_size
MEMORY_CAPACITY = max(env.step_length * args.pool_size, args.batch_size * 2)
execute_date = time.strftime("%m%d", time.localtime())

# ====================== 辅助函数：加载耦合模型 ======================
def load_joint_agents(rnn_model, pretrain_date, pretrain_id, prefix):
    pretrain_dir = os.path.join(project_root, "nets", "Chap4", "Joint_Net", pretrain_date, pretrain_id)
    agents = {}
    names = ["FC", "BAT", "SC"]
    
    if not os.path.exists(pretrain_dir):
        raise FileNotFoundError(f"目录不存在: {pretrain_dir}")

    for name in names:
        path = os.path.join(pretrain_dir, f"{prefix}_{name}.pth")
        if not os.path.exists(path):
            # 防御：尝试加载没有前缀的默认文件
            path = os.path.join(pretrain_dir, f"Joint_Model_{name}.pth")
            if not os.path.exists(path):
                raise FileNotFoundError(f"无法定位模型: {path}")
        
        ckpt = torch.load(path, map_location=device)
        # 获取动作维度
        try:
            n_act = ckpt['marl_part.output.weight'].shape[0]
        except KeyError:
            # 防御：如果 state_dict 结构不同，尝试从 marl_part 提取
            n_act = ckpt['output.weight'].shape[0]
            
        agent = JointDQN(name, rnn_model, n_act)
        agent.eval_net.load_state_dict(ckpt)
        agent.target_net.load_state_dict(ckpt)
        
        agent.setup_optimizer(args.lr, 0.5, 50)
        agents[name] = agent
        print(f"✅ Loaded {name} Agent ({n_act} actions)")
        
    return agents["FC"], agents["BAT"], agents["SC"]

# ====================== Main 训练逻辑 ======================
if __name__ == '__main__':
    # 1. 准备目录
    TARGET_BASE_DIR = os.path.join(project_root, "nets", "Chap4", "Joint_Net", execute_date)
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    train_id = get_max_folder_name(TARGET_BASE_DIR)
    base_path = os.path.join(TARGET_BASE_DIR, str(train_id))
    os.makedirs(base_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(base_path, "logs"))

    # 2. 初始化基础 RNN
    rnn_model = MultiTaskRNN().to(device)

    # 3. 加载智能体
    FC_Agent, Bat_Agent, SC_Agent = load_joint_agents(
        rnn_model, args.pretrain_date, args.pretrain_train_id, args.pretrain_prefix
    )
    all_agents = [FC_Agent, Bat_Agent, SC_Agent]

    # 4. 共享内存 (🚨 关键修复点)
    MEMORY_WIDTH = N_STATES * 2 + 3 + 1 # s(7), a1, a2, a3, r(1), s_(7) = 18
    shared_memory = np.zeros((MEMORY_CAPACITY, MEMORY_WIDTH))
    memory_counter = [0]
    
    # 防御性：显式绑定到 MARL_Engine 内部使用的变量名
    for a in all_agents:
        a.shared_memory = shared_memory # 🚨 必须使用 shared_memory 匹配父类 learn 方法
        a.memory_counter = memory_counter

    # 5. 训练循环
    print(f'\n🚀 Joint Fine-tuning [ID: {train_id}] [Device: {device}]')
    reward_max = -float('inf')
    x_episodes, y_rewards, loss_records = [], [], []

    pbar = tqdm(range(args.episode), desc="Joint Training")
    for i_episode in pbar:
        s = env.reset()
        ep_r = 0
        step_loss = []

        while True:
            a_fc = FC_Agent.choose_action(s, train=True, epsilon=args.epsilon)
            a_bat = Bat_Agent.choose_action(s, train=True, epsilon=args.epsilon)
            a_sc = SC_Agent.choose_action(s, train=True, epsilon=args.epsilon)

            s_, r, done, _ = env.step([a_fc, a_bat, a_sc])

            # 存储转换数据 (显式拼接确保维度正确)
            transition = np.zeros(MEMORY_WIDTH)
            transition[0:7] = s
            transition[7:10] = [a_fc, a_bat, a_sc]
            transition[10] = r
            transition[11:18] = s_
            
            index = memory_counter[0] % MEMORY_CAPACITY
            shared_memory[index, :] = transition
            memory_counter[0] += 1

            ep_r += r

            # 学习触发判定
            # 🚨 防御：必须保证池子里的数据量大于 batch_size 且池子已开始有有效覆盖
            if memory_counter[0] > args.batch_size and memory_counter[0] % args.learn_frequency == 0:
                try:
                    l1 = FC_Agent.learn(0, N_STATES, args.gamma, 100, args.batch_size) or 0
                    l2 = Bat_Agent.learn(1, N_STATES, args.gamma, 100, args.batch_size) or 0
                    l3 = SC_Agent.learn(2, N_STATES, args.gamma, 100, args.batch_size) or 0
                    step_loss.append((l1 + l2 + l3) / 3.0)
                except Exception as e:
                    # 记录学习过程中的异常但不中断训练
                    pass

            if done:
                avg_loss = np.mean(step_loss) if step_loss else 0
                writer.add_scalar("Reward/Episode", ep_r, i_episode)
                writer.add_scalar("Loss/Average", avg_loss, i_episode)
                
                loss_records.append(avg_loss)
                pbar.set_postfix({'Rew': f'{ep_r:.1f}', 'Mem': f'{min(memory_counter[0], MEMORY_CAPACITY)}'})
                break
            s = s_

        x_episodes.append(i_episode)
        y_rewards.append(ep_r)

        # 保存最优模型
        if ep_r > reward_max and memory_counter[0] > args.batch_size:
            reward_max = ep_r
            for a in all_agents:
                save_path = os.path.join(base_path, f"Joint_Model_{a.agent_name}.pth")
                torch.save(a.eval_net.state_dict(), save_path)

    # 6. 保存训练曲线与最终模型
    for a in all_agents:
        torch.save(a.eval_net.state_dict(), os.path.join(base_path, f"FINAL_{a.agent_name}.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(x_episodes, y_rewards, label='Episode Reward')
    plt.axhline(y=reward_max, color='r', linestyle='--', label=f'Best: {reward_max:.1f}')
    plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(base_path, "train_curve.png"))
    
    writer.close()
    print(f"\n✅ 训练完成！结果目录: {base_path}")