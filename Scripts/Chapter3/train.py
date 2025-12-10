import os
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
# 导入公共模块
from MARL_Engine import (
    setup_project_root, device, 
    IndependentDQN, get_max_folder_name,
    font_get
)
project_root = setup_project_root()
from Scripts.Env import Envs

# 获取新罗马
font_get()

# 全局设置与超参数
env = Envs()
writer = SummaryWriter()
torch.set_default_dtype(torch.float32)

# 超参数
BATCH_SIZE = 32
LR = 0.002
EPSILON = 0.9
GAMMA = 0.95
TARGET_REPLACE_ITER = 100
POOL_SIZE = 10
EPISODE = 200
LEARN_FREQUENCY = 10
REAL_TIME_DRAW = False

# 学习率调度与早停参数
LR_PATIENCE = 50
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 100
REWARD_THRESHOLD = 0.001

# 环境参数
N_STATES = env.observation_space.shape[0]
N_TOTAL_ACTIONS = env.N_ACTIONS
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2

# 内存配置
MEMORY_CAPACITY = env.step_length * POOL_SIZE
current_timestamp = time.time()
local_time = time.localtime(current_timestamp)
execute_date = time.strftime("%m%d", local_time)
remark = "MARL_IQL_32x20x2"

# 验证动作分解
N_EXPECTED_ACTIONS = N_FC_ACTIONS * N_BAT_ACTIONS * N_SC_ACTIONS
if N_EXPECTED_ACTIONS != N_TOTAL_ACTIONS:
    print(f"警告：动作分解 {N_EXPECTED_ACTIONS} 与环境 N_TOTAL_ACTIONS({N_TOTAL_ACTIONS}) 不匹配")

# 时间分解打印函数
def print_time_breakdown(episode, episode_times):
    total_time = sum(episode_times.values())
    if total_time < 1e-6:
        print(f"回合 {episode} 耗时过短，跳过耗时分析。")
        return

    print("\n" + "=" * 45)
    print(f"🚀 回合 {episode} 耗时分解 (总耗时: {total_time:.4f} s)")
    print("-" * 45)
    for name, time_val in sorted(episode_times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_val / total_time) * 100
        print(f"| {name.ljust(15)} | {time_val:9.4f} s | {percentage:6.2f} % |")
    print("=" * 45)

if __name__ == '__main__':
    # 路径设置
    TARGET_BASE_DIR = os.path.join(project_root, "nets", "Chap3", execute_date)
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    train_id = get_max_folder_name(TARGET_BASE_DIR)
    base_path = f"{TARGET_BASE_DIR}/{train_id}"
    os.makedirs(base_path)

    # 共享内存初始化
    MEMORY_WIDTH = N_STATES * 2 + 4
    shared_memory = np.zeros((MEMORY_CAPACITY, MEMORY_WIDTH))
    memory_counter = [0]

    # 初始化智能体
    FC_Agent = IndependentDQN(
        "FC_Agent", N_STATES, N_FC_ACTIONS, 
        shared_memory, memory_counter
    )
    Bat_Agent = IndependentDQN(
        "Bat_Agent", N_STATES, N_BAT_ACTIONS, 
        shared_memory, memory_counter
    )
    SC_Agent = IndependentDQN(
        "SC_Agent", N_STATES, N_SC_ACTIONS, 
        shared_memory, memory_counter
    )
    all_agents = [FC_Agent, Bat_Agent, SC_Agent]

    # 设置优化器
    for agent in all_agents:
        agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)

    # 训练过程
    print('\nCollecting experience and learning (I-DQN, 3-Agent)...')
    start_time_total = time.time()
    reward_max = -float('inf')
    reward_not_improve_episodes = 0
    training_done = False
    x, y = [], []

    if REAL_TIME_DRAW:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(x, y)

    episode_pbar = tqdm(range(EPISODE), desc=f"RL Training ({remark})")
    for i_episode in episode_pbar:
        if training_done:
            break

        s = env.reset()
        ep_r = 0
        episode_times = {
            'Action_Select': 0.0,
            'Env_Step': 0.0,
            'Data_Store': 0.0,
            'DQN_Learn': 0.0
        }
        step_count = 0

        while True:
            # 动作选择
            time_start_action = time.time()
            a_fc = FC_Agent.choose_action(s, train=True, epsilon=EPSILON)
            a_bat = Bat_Agent.choose_action(s, train=True, epsilon=EPSILON)
            a_sc = SC_Agent.choose_action(s, train=True, epsilon=EPSILON)
            episode_times['Action_Select'] += (time.time() - time_start_action)

            # 环境交互
            action_list = [a_fc, a_bat, a_sc]
            time_start_step = time.time()
            s_, r, done, _ = env.step(action_list)
            episode_times['Env_Step'] += (time.time() - time_start_step)

            # 存储转换
            time_start_store = time.time()
            transition = np.hstack((s, a_fc, a_bat, a_sc, r, s_))
            index = memory_counter[0] % MEMORY_CAPACITY
            if transition.shape[0] != MEMORY_WIDTH:
                raise RuntimeError(f"存储转换长度错误: 期望 {MEMORY_WIDTH}, 实际 {transition.shape[0]}")
            shared_memory[index, :] = transition
            memory_counter[0] += 1
            episode_times['Data_Store'] += (time.time() - time_start_store)

            ep_r += r
            step_count += 1

            # 学习过程
            if memory_counter[0] > MEMORY_CAPACITY and memory_counter[0] % LEARN_FREQUENCY == 0:
                time_start_learn = time.time()
                FC_Agent.learn(0, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE)
                Bat_Agent.learn(1, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE)
                SC_Agent.learn(2, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE)
                episode_times['DQN_Learn'] += (time.time() - time_start_learn)

            if done:
                writer.add_scalar("Ep_r/Ep", ep_r, i_episode)
                using_time_total = time.time() - start_time_total
                current_lr = FC_Agent.optimizer.param_groups[0]["lr"]
                episode_pbar.set_postfix({
                    'Ep_r': f'{ep_r:.2f}',
                    'LR': f'{current_lr:.2e}',
                    'Total_Time': f'{using_time_total:.2f}s',
                })

                if i_episode < 2 or (i_episode + 1) % 100 == 0:
                    print_time_breakdown(i_episode + 1, episode_times)
                break

            s = s_

        x.append(i_episode)
        y.append(ep_r)

        # 模型保存与早停逻辑
        if ep_r > reward_max + REWARD_THRESHOLD:
            reward_max = ep_r
            reward_not_improve_episodes = 0
            net_name_base = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR*10000)}_ep_{i_episode+1}"
                           f"_pool{POOL_SIZE}_freq{LEARN_FREQUENCY}_MARL_{remark}_MAX_R{int(reward_max)}")
            torch.save(FC_Agent.eval_net.state_dict(), f"{net_name_base}_FC.pth")
            torch.save(Bat_Agent.eval_net.state_dict(), f"{net_name_base}_BAT.pth")
            torch.save(SC_Agent.eval_net.state_dict(), f"{net_name_base}_SC.pth")
            print(f"\n--- New Max Reward: {reward_max:.2f} | Models saved: {net_name_base} ---")
        else:
            reward_not_improve_episodes += 1

        # 学习率调度
        for agent in all_agents:
            agent.scheduler.step(ep_r)

        # 早停检查
        if reward_not_improve_episodes >= EARLY_STOP_PATIENCE:
            print(f"\n--- Early Stopping Triggered! ---")
            training_done = True

    # 最终处理
    final_episode = i_episode + 1 if not training_done else i_episode
    final_net_name_base = (f"{base_path}/final_bs{BATCH_SIZE}_lr{int(LR*10000)}_ep_{final_episode}_pool{POOL_SIZE}"
                         f"_freq{LEARN_FREQUENCY}_MARL_{remark}_FINAL")
    torch.save(FC_Agent.eval_net.state_dict(), f"{final_net_name_base}_FC.pth")
    torch.save(Bat_Agent.eval_net.state_dict(), f"{final_net_name_base}_BAT.pth")
    torch.save(SC_Agent.eval_net.state_dict(), f"{final_net_name_base}_SC.pth")
    print(f"\nFinal models saved: {final_net_name_base}")

    # 可视化与保存
    writer.flush()
    writer.close()
    plt.figure()
    plt.plot(x, y)
    plt.savefig(f"{base_path}/train_curve_MARL_IQL_bs{BATCH_SIZE}_lr{int(LR*10000)}_ep{final_episode}.svg")
    if REAL_TIME_DRAW:
        plt.ioff()