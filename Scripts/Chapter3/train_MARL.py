import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau  # ❗ 新增：学习率调度器

# --------------------------------------------------------------------
# 导入路径修正：将项目根目录添加到 sys.path
# --------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------------------------

from Scripts.Env import Envs

# ====================================================================
# 全局设置与超参数
# ====================================================================
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = Envs()
writer = SummaryWriter()
torch.set_default_dtype(torch.float32)

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.001
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
POOL_SIZE = 10
EPISODE = 2000
LEARN_FREQUENCY = 10

REAL_TIME_DRAW = False

# --------------------------------------------------------------------
# ❗ 新增：RL 和 Early Stopping 超参数
# --------------------------------------------------------------------
# RL 学习率调度器参数 (基于奖励)
LR_PATIENCE = 50  # 奖励连续多少个 episode 没有显著提高时触发 LR 减小
LR_FACTOR = 0.5  # 触发时 LR 减小的比例
# 早停参数 (基于奖励)
EARLY_STOP_PATIENCE = 100  # 奖励连续多少个 episode 没有显著提高时触发早停
REWARD_THRESHOLD = 0.001  # 判定奖励提高的阈值 (绝对值)
# --------------------------------------------------------------------


MEMORY_CAPACITY = env.step_length * POOL_SIZE
current_timestamp = time.time()
local_time = time.localtime(current_timestamp)
execute_date = time.strftime("%m%d", local_time)
remark = "MARL_IQL_32x20x2"

# Environment Constants
N_STATES = env.observation_space.shape[0]
N_TOTAL_ACTIONS = env.N_ACTIONS

# MARL 动作分解修正
N_FC_ACTIONS = 32
N_BAT_ACTIONS = 20
N_SC_ACTIONS = 2

N_EXPECTED_ACTIONS = N_FC_ACTIONS * N_BAT_ACTIONS * N_SC_ACTIONS

if N_EXPECTED_ACTIONS != N_TOTAL_ACTIONS:
    print(
        f"警告：动作分解 {N_EXPECTED_ACTIONS} 与环境 N_TOTAL_ACTIONS({N_TOTAL_ACTIONS}) 不匹配。代码将继续运行，但请检查 Env.py。")
    pass

Base_Model_Name = ""


class Net(nn.Module):
    def __init__(self, N_ACTIONS):
        torch.manual_seed(0)
        super(Net, self).__init__()
        self.input = nn.Linear(N_STATES, 64)
        self.input.weight.data.normal_(0, 0.1)

        self.lay1 = nn.Linear(64, 64)
        self.lay1.weight.data.normal_(0, 0.1)

        self.output = nn.Linear(64, N_ACTIONS)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.lay1(x)
        x = F.relu(x)
        actions_value = self.output(x)
        return actions_value


class IndependentDQN(object):
    def __init__(self, agent_name, N_AGENT_ACTIONS, shared_memory, memory_counter_ref):

        self.agent_name = agent_name
        self.N_AGENT_ACTIONS = N_AGENT_ACTIONS

        self.eval_net = Net(N_AGENT_ACTIONS).to(device)
        self.target_net = Net(N_AGENT_ACTIONS).to(device)

        self.learn_step_counter = 0
        self.memory = shared_memory
        self.memory_counter_ref = memory_counter_ref

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # ❗ 实例化 ReduceLROnPlateau 调度器，监控奖励值，mode='max'
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode='max',
                                           factor=LR_FACTOR,
                                           patience=LR_PATIENCE,
                                        #    verbose=True,
                                           min_lr=1e-6)

    def load_net(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.eval_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state_input: torch.Tensor, train=True):
        temp = torch.FloatTensor(state_input)
        state_input = torch.unsqueeze(temp.to(device), 0)

        epsilon = 1.0 if train else EPSILON

        if np.random.uniform() < epsilon:
            with torch.no_grad():
                actions_value = self.eval_net.forward(state_input)
                action_index = torch.max(actions_value, 1)[1].item()
        else:
            action_index = np.random.randint(0, self.N_AGENT_ACTIONS)

        return action_index

    def learn(self, agent_idx):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions from shared memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)

        action_column_index = N_STATES + agent_idx
        b_a = torch.LongTensor(b_memory[:, action_column_index:action_column_index + 1].astype(int)).to(device)

        b_r = torch.FloatTensor(b_memory[:, N_STATES + 3:N_STATES + 4]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, N_STATES + 4:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_max_folder_name(directory):
    if not os.path.exists(directory):
        return 0
    folders = [int(name) for name in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, name)) and name.isdigit()]
    if not folders:
        return 0
    return max(folders) + 1


# --------------------------------------------------------------------
def print_time_breakdown(episode, episode_times):
    """打印本回合的耗时分解结果"""
    total_time = sum(episode_times.values())
    if total_time < 1e-6:
        print(f"回合 {episode} 耗时过短，跳过耗时分析。")
        return

    print("\n" + "=" * 45)
    print(f"🚀 回合 {episode} 耗时分解 (总耗时: {total_time:.4f} s)")
    print("-" * 45)

    sorted_times = sorted(episode_times.items(), key=lambda item: item[1], reverse=True)
    for name, time_val in sorted_times:
        percentage = (time_val / total_time) * 100
        print(f"| {name.ljust(15)} | {time_val:9.4f} s | {percentage:6.2f} % |")

    print("=" * 45)


# --------------------------------------------------------------------


if __name__ == '__main__':
    # --------------------------------------------------------------------
    # 路径和日志设置
    # --------------------------------------------------------------------
    TARGET_BASE_DIR = os.path.join(project_root, "nets", "Chap3", execute_date)
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    train_id = get_max_folder_name(TARGET_BASE_DIR)
    base_path = f"{TARGET_BASE_DIR}/{train_id}"
    os.makedirs(base_path)

    # 初始化共享内存和计数器
    MEMORY_WIDTH = N_STATES * 2 + 4
    shared_memory = np.zeros((MEMORY_CAPACITY, MEMORY_WIDTH))
    memory_counter = [0]

    # 实例化三个独立的 DQN 智能体
    FC_Agent = IndependentDQN("FC_Agent", N_FC_ACTIONS, shared_memory, memory_counter)
    Bat_Agent = IndependentDQN("Bat_Agent", N_BAT_ACTIONS, shared_memory, memory_counter)
    SC_Agent = IndependentDQN("SC_Agent", N_SC_ACTIONS, shared_memory, memory_counter)

    # 将所有智能体放在一个列表中，方便统一操作 (LR 调度和早停)
    all_agents = [FC_Agent, Bat_Agent, SC_Agent]

    print('\nCollecting experience and learning (I-DQN, 3-Agent)...')
    start_time_total = time.time()

    # ❗ 增加 RL 和 Early Stopping 变量
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

        # ❗ 早停检查，如果满足条件则退出训练循环
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
            # 1. 动作选择
            time_start_action = time.time()
            a_fc = FC_Agent.choose_action(s)
            a_bat = Bat_Agent.choose_action(s)
            a_sc = SC_Agent.choose_action(s)
            episode_times['Action_Select'] += (time.time() - time_start_action)

            # 2. 环境交互
            action_list = [a_fc, a_bat, a_sc]
            time_start_step = time.time()
            s_, r, done, _ = env.step(action_list)
            episode_times['Env_Step'] += (time.time() - time_start_step)

            # 3. 存储转换
            time_start_store = time.time()
            transition = np.hstack((s, a_fc, a_bat, a_sc, r, s_))

            index = memory_counter[0] % MEMORY_CAPACITY
            if transition.shape[0] != MEMORY_WIDTH:
                raise RuntimeError(
                    f"存储转换长度错误: 期望 {MEMORY_WIDTH}, 实际 {transition.shape[0]}. 请检查 N_STATES 和动作分解是否正确。")

            shared_memory[index, :] = transition
            memory_counter[0] += 1
            episode_times['Data_Store'] += (time.time() - time_start_store)

            ep_r += r
            step_count += 1

            # 4. I-DQN 独立学习
            if memory_counter[0] > MEMORY_CAPACITY and memory_counter[0] % LEARN_FREQUENCY == 0:
                time_start_learn = time.time()
                FC_Agent.learn(0)
                Bat_Agent.learn(1)
                SC_Agent.learn(2)
                episode_times['DQN_Learn'] += (time.time() - time_start_learn)

            if done:
                writer.add_scalar("Ep_r/Ep", ep_r, i_episode)
                using_time_total = time.time() - start_time_total

                current_lr = FC_Agent.optimizer.param_groups[0]["lr"]  # 获取当前LR

                # 更新进度条信息
                episode_pbar.set_postfix({
                    'Ep_r': f'{ep_r:.2f}',
                    'LR': f'{current_lr:.2e}',
                    'Total_Time': f'{using_time_total:.2f}s',
                })

                if i_episode < 2 or (i_episode + 1) % 100 == 0:
                    print_time_breakdown(i_episode + 1, episode_times)

                break

            s = s_

        x.append(int(i_episode))
        y.append(float(ep_r))

        # --------------------------------------------------------
        # ❗ 模型保存 (最高奖励) / RL / 早停逻辑 (基于回合奖励)
        # --------------------------------------------------------

        # 1. 模型保存 (仅保留回合奖励最高的模型)
        if ep_r > reward_max + REWARD_THRESHOLD:  # 奖励显著提高
            reward_max = ep_r
            reward_not_improve_episodes = 0  # 奖励提高，重置计数器

            # 保存所有三个智能体的最高奖励模型
            net_name_base = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_ep_{i_episode + 1}"
                             f"_pool{POOL_SIZE}_freq{LEARN_FREQUENCY}_MARL_{remark}_MAX_R{int(reward_max)}")
            torch.save(FC_Agent.eval_net.state_dict(), f"{net_name_base}_FC.pth")
            torch.save(Bat_Agent.eval_net.state_dict(), f"{net_name_base}_BAT.pth")
            torch.save(SC_Agent.eval_net.state_dict(), f"{net_name_base}_SC.pth")
            print(f"\n--- New Max Reward: {reward_max:.2f} | Models saved: {net_name_base} ---")

            if REAL_TIME_DRAW:
                ax.plot(i_episode, reward_max, 'ro')

        # 2. 早停机制 (Early Stopping)
        elif ep_r <= reward_max + REWARD_THRESHOLD:  # 奖励没有显著提高
            reward_not_improve_episodes += 1  # ❗ 已修正缩进问题

        # 3. RL 学习率调度 (对所有智能体应用)
        for agent in all_agents:
            agent.scheduler.step(ep_r)

    if reward_not_improve_episodes >= EARLY_STOP_PATIENCE:
        print(f"\n\n--- Early Stopping Triggered! ---")
        print(f"Reward has not improved significantly for {EARLY_STOP_PATIENCE} episodes.")
        training_done = True

        # --------------------------------------------------------------------

    if REAL_TIME_DRAW:
        line.set_xdata(x)
        line.set_ydata(y)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

# 最终模型保存 (在训练循环结束后)
final_episode = i_episode + 1 if not training_done else i_episode  # 记录实际训练的回合数
final_net_name_base = (f"{base_path}/final_bs{BATCH_SIZE}_lr{int(LR * 10000)}_ep_{final_episode}_pool{POOL_SIZE}"
                       f"_freq{LEARN_FREQUENCY}_MARL_{remark}_FINAL")

torch.save(FC_Agent.eval_net.state_dict(), f"{final_net_name_base}_FC.pth")
torch.save(Bat_Agent.eval_net.state_dict(), f"{final_net_name_base}_BAT.pth")
torch.save(SC_Agent.eval_net.state_dict(), f"{final_net_name_base}_SC.pth")
print(f"\nFinal models saved: {final_net_name_base}")

writer.flush()
writer.close()

# 绘制并保存曲线
if not REAL_TIME_DRAW:
    fig, ax = plt.subplots()
    ax.plot(x, y)

# 绘制并保存曲线
plt.savefig(f"{base_path}/train_curve_MARL_IQL_bs{BATCH_SIZE}_lr{int(LR * 10000)}_ep{final_episode}.svg")

if REAL_TIME_DRAW:
    plt.ioff()
plt.show()