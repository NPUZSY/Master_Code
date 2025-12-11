import os
import time
import json
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
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
POOL_SIZE = 100
EPISODE = 1000
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
execute_time = time.strftime("%H%M%S", local_time)  # 新增：记录具体时间
remark = "MARL_IQL_32x20x2"

# 新增：全局变量存储最优模型文件名
best_model_base_name = ""

# 验证动作分解
N_EXPECTED_ACTIONS = N_FC_ACTIONS * N_BAT_ACTIONS * N_SC_ACTIONS
if N_EXPECTED_ACTIONS != N_TOTAL_ACTIONS:
    print(f"警告：动作分解 {N_EXPECTED_ACTIONS} 与环境 N_TOTAL_ACTIONS({N_TOTAL_ACTIONS}) 不匹配")

# 新增：定义保存超参数的函数
def save_hyperparameters(save_path, final_metrics=None):
    """
    保存超参数到指定路径（txt和json格式）
    :param save_path: 保存目录
    :param final_metrics: 训练最终指标（如最大奖励、最终奖励等）
    """
    # 整理超参数字典
    hyperparams = {
        # 基础信息
        "train_info": {
            "execute_date": execute_date,
            "execute_time": execute_time,
            "train_id": os.path.basename(save_path),
            "remark": remark,
            "device": str(device),
            "total_training_time_s": round(time.time() - start_time_total, 2) if 'start_time_total' in globals() else 0,
            "best_model_base_name": best_model_base_name,  # 新增：最优模型文件名前缀
            "best_model_full_path": os.path.join(save_path, best_model_base_name) if best_model_base_name else ""  # 新增：最优模型完整路径
        },
        # 核心超参数
        "core_hyperparams": {
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "EPSILON": EPSILON,
            "GAMMA": GAMMA,
            "TARGET_REPLACE_ITER": TARGET_REPLACE_ITER,
            "POOL_SIZE": POOL_SIZE,
            "EPISODE": EPISODE,
            "LEARN_FREQUENCY": LEARN_FREQUENCY,
            "MEMORY_CAPACITY": MEMORY_CAPACITY,
            "REAL_TIME_DRAW": REAL_TIME_DRAW
        },
        # 学习率调度与早停参数
        "lr_earlystop_params": {
            "LR_PATIENCE": LR_PATIENCE,
            "LR_FACTOR": LR_FACTOR,
            "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
            "REWARD_THRESHOLD": REWARD_THRESHOLD
        },
        # 环境参数
        "env_params": {
            "N_STATES": N_STATES,
            "N_TOTAL_ACTIONS": N_TOTAL_ACTIONS,
            "N_FC_ACTIONS": N_FC_ACTIONS,
            "N_BAT_ACTIONS": N_BAT_ACTIONS,
            "N_SC_ACTIONS": N_SC_ACTIONS,
            "N_EXPECTED_ACTIONS": N_EXPECTED_ACTIONS,
            "step_length": env.step_length if hasattr(env, 'step_length') else "unknown"
        },
        # 训练结果指标
        "training_metrics": final_metrics or {}
    }

    # 保存为JSON格式（便于程序解析）
    json_path = os.path.join(save_path, "hyperparameters.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

    # 保存为TXT格式（便于人工阅读）
    txt_path = os.path.join(save_path, "hyperparameters.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                训练超参数汇总                \n")
        f.write("=" * 80 + "\n\n")
        
        for section, params in hyperparams.items():
            f.write(f"【{section.upper()}】\n")
            f.write("-" * 60 + "\n")
            for key, value in params.items():
                # 对最优模型名称单独高亮显示
                if key in ["best_model_base_name", "best_model_full_path"]:
                    f.write(f"{key:<30}: \033[1;32m{value}\033[0m\n")  # 绿色高亮
                else:
                    f.write(f"{key:<30}: {value}\n")
            f.write("\n")

    # 单独打印最优模型名称（方便直接复制）
    if best_model_base_name:
        print(f"\n🎯 最优模型文件名前缀（可直接复制）：")
        print(f"   {best_model_base_name}")
    print(f"\n✅ 超参数已保存到：")
    print(f"   JSON格式: {json_path}")
    print(f"   TXT格式: {txt_path}")

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
            best_model_base_name = (f"bs{BATCH_SIZE}_lr{int(LR*10000)}_ep_{i_episode+1}"  # 新增：更新全局最优模型名称
                                   f"_pool{POOL_SIZE}_freq{LEARN_FREQUENCY}_MARL_{remark}_MAX_R{int(reward_max)}")
            net_name_base = best_model_base_name  # 保持原有逻辑
            torch.save(FC_Agent.eval_net.state_dict(), f"{base_path}/{net_name_base}_FC.pth")
            torch.save(Bat_Agent.eval_net.state_dict(), f"{base_path}/{net_name_base}_BAT.pth")
            torch.save(SC_Agent.eval_net.state_dict(), f"{base_path}/{net_name_base}_SC.pth")
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

    # 新增：整理训练最终指标
    final_metrics = {
        "max_reward": round(reward_max, 4),
        "final_reward": round(y[-1], 4) if y else 0,
        "average_reward": round(np.mean(y) if y else 0, 4),
        "total_episodes_completed": final_episode,
        "early_stopped": training_done,
        "final_learning_rate": round(FC_Agent.optimizer.param_groups[0]["lr"], 6),
        "reward_not_improve_episodes": reward_not_improve_episodes,
        "best_model_reward": round(reward_max, 4)  # 新增：最优模型对应的奖励
    }

    # 新增：保存超参数
    save_hyperparameters(base_path, final_metrics)

    # 可视化与保存
    writer.flush()
    writer.close()
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title(f'Training Curve (MARL_IQL, Ep={final_episode})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{base_path}/train_curve_MARL_IQL_bs{BATCH_SIZE}_lr{int(LR*10000)}_ep{final_episode}.svg", 
                bbox_inches='tight', dpi=300)
    if REAL_TIME_DRAW:
        plt.ioff()
        plt.show()

    print(f"\n🎉 训练完成！所有文件已保存到: {base_path}")
    # 最终再次打印最优模型名称（方便复制）
    if best_model_base_name:
        print(f"\n📋 最优模型文件名前缀（直接复制即可）：")
        print(f"{best_model_base_name}")