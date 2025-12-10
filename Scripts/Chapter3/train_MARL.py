import os
import sys  # ❗ 引入 sys 模块

# --------------------------------------------------------------------
# 导入路径修正：将项目根目录添加到 sys.path
# --------------------------------------------------------------------
# 获取当前文件（train_MARL.py）的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级，得到项目根目录: E:\Master\毕业\硕士毕业论文代码仓库 (假设您的项目结构)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# 将项目根目录添加到 Python 搜索路径中
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from tqdm import tqdm

# 修正：现在可以正确地通过 Scripts 找到 Env 模块
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

# Hyper Parameters (保持原设定)
BATCH_SIZE = 64
LR = 0.005
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
POOL_SIZE = 10
EPISODE = 1000
# 学习频率
LEARN_FREQUENCY = 10

REAL_TIME_DRAW = False

MEMORY_CAPACITY = env.step_length * POOL_SIZE
current_timestamp = time.time()
local_time = time.localtime(current_timestamp)
execute_date = time.strftime("%m%d", local_time)
remark = "MARL_IQL_32x20x2"

# Environment Constants
N_STATES = env.observation_space.shape[0]
N_TOTAL_ACTIONS = env.N_ACTIONS

# --------------------------------------------------------------------
# MARL 动作分解修正 (Action Decomposition for I-DQN)
# --------------------------------------------------------------------
N_FC_ACTIONS = 32  # FC 功率变化率 (32 份)
N_BAT_ACTIONS = 20  # Battery 输出功率 (20 份)
N_SC_ACTIONS = 2  # SuperCap 状态 (切入/切出)

N_EXPECTED_ACTIONS = N_FC_ACTIONS * N_BAT_ACTIONS * N_SC_ACTIONS

if N_EXPECTED_ACTIONS != N_TOTAL_ACTIONS:
    print(
        f"警告：动作分解 {N_EXPECTED_ACTIONS} 与环境 N_TOTAL_ACTIONS({N_TOTAL_ACTIONS}) 不匹配。代码将继续运行，但请检查 Env.py。")
    pass  # 允许不匹配继续运行，但用户应确保环境动作空间正确

Base_Model_Name = ""


class Net(nn.Module):
    """
    通用 Q-网络结构。输入全局状态 N_STATES，输出各自局部动作空间 N_ACTIONS。
    """

    def __init__(self, N_ACTIONS):
        torch.manual_seed(0)
        super(Net, self).__init__()
        # 使用更大的网络层以适应更大的动作空间
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
    """
    Independent DQN (I-DQN) 智能体类
    """

    def __init__(self, agent_name, N_AGENT_ACTIONS, shared_memory, memory_counter_ref):

        self.agent_name = agent_name
        self.N_AGENT_ACTIONS = N_AGENT_ACTIONS

        # 使用局部动作空间大小初始化网络
        self.eval_net = Net(N_AGENT_ACTIONS).to(device)
        self.target_net = Net(N_AGENT_ACTIONS).to(device)

        self.learn_step_counter = 0
        self.memory = shared_memory  # 引用共享内存
        self.memory_counter_ref = memory_counter_ref  # 引用内存计数器

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def load_net(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=device))
        self.eval_net.to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state_input: torch.Tensor, train=True):
        temp = torch.FloatTensor(state_input)
        state_input = torch.unsqueeze(temp.to(device), 0)

        # 策略：训练初期随机探索，后期ε-greedy
        epsilon = 1.0 if train else EPSILON

        if np.random.uniform() < epsilon:  # greedy
            with torch.no_grad():
                actions_value = self.eval_net.forward(state_input)
                # 选择 Q 值最大的局部动作索引
                action_index = torch.max(actions_value, 1)[1].item()
        else:  # random
            action_index = np.random.randint(0, self.N_AGENT_ACTIONS)

        return action_index

    # learn 方法现在接受 agent_idx 来索引共享内存中的局部动作
    def learn(self, agent_idx):
        memory_counter = self.memory_counter_ref[0]

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions from shared memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)

        # 局部动作索引：FC=0, Bat=1, SC=2
        action_column_index = N_STATES + agent_idx
        b_a = torch.LongTensor(b_memory[:, action_column_index:action_column_index + 1].astype(int)).to(device)

        # 奖励在 N_STATES + 3 处 (因为有 3 个动作列)
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 3:N_STATES + 4]).to(device)
        # s' 在 N_STATES + 4 处开始
        b_s_ = torch.FloatTensor(b_memory[:, N_STATES + 4:]).to(device)

        # I-DQN Q-target 计算
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
    # 过滤出目录名且为数字的文件夹
    folders = [int(name) for name in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, name)) and name.isdigit()]
    if not folders:
        return 0
    return max(folders) + 1


# --------------------------------------------------------------------
# ❗ 新增：用于打印耗时分析的辅助函数
# --------------------------------------------------------------------
def print_time_breakdown(episode, episode_times):
    """打印本回合的耗时分解结果"""
    total_time = sum(episode_times.values())

    # 防止除零
    if total_time < 1e-6:
        print(f"回合 {episode} 耗时过短，跳过耗时分析。")
        return

    print("\n" + "=" * 45)
    print(f"🚀 回合 {episode} 耗时分解 (总耗时: {total_time:.4f} s)")
    print("-" * 45)

    # 打印每个部分的耗时及其占总时间的百分比
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

    # 自动获取下一个训练ID
    train_id = get_max_folder_name(TARGET_BASE_DIR)

    # 最终的保存路径
    base_path = f"{TARGET_BASE_DIR}/{train_id}"
    os.makedirs(base_path)

    # 初始化共享内存和计数器
    # 内存存储结构: [s, a_fc, a_bat, a_sc, r, s_] (N_STATES * 2 + 4)
    MEMORY_WIDTH = N_STATES * 2 + 4
    shared_memory = np.zeros((MEMORY_CAPACITY, MEMORY_WIDTH))
    memory_counter = [0]  # 使用列表作为可变引用，以便在类中更新

    # 实例化三个独立的 DQN 智能体
    FC_Agent = IndependentDQN("FC_Agent", N_FC_ACTIONS, shared_memory, memory_counter)
    Bat_Agent = IndependentDQN("Bat_Agent", N_BAT_ACTIONS, shared_memory, memory_counter)
    SC_Agent = IndependentDQN("SC_Agent", N_SC_ACTIONS, shared_memory, memory_counter)

    print('\nCollecting experience and learning (I-DQN, 3-Agent)...')
    start_time_total = time.time()
    reward_max = -1e6
    x, y = [], []

    if REAL_TIME_DRAW:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(x, y)

    # 使用 tqdm 包装主循环，实现实时进度输出
    episode_pbar = tqdm(range(EPISODE), desc=f"RL Training ({remark})")

    for i_episode in episode_pbar:
        s = env.reset()
        ep_r = 0

        # ❗ 初始化本回合的耗时追踪器
        episode_times = {
            'Action_Select': 0.0,
            'Env_Step': 0.0,
            'Data_Store': 0.0,
            'DQN_Learn': 0.0
        }

        step_count = 0

        while True:
            # --------------------------------------------------------
            # 1. 动作选择 (Action Selection)
            # --------------------------------------------------------
            time_start_action = time.time()
            a_fc = FC_Agent.choose_action(s)  # FC 局部动作索引 a_fc ∈ {0, ..., 31}
            a_bat = Bat_Agent.choose_action(s)  # Bat 局部动作索引 a_bat ∈ {0, ..., 19}
            a_sc = SC_Agent.choose_action(s)  # SC 局部动作索引 a_sc ∈ {0, 1}
            episode_times['Action_Select'] += (time.time() - time_start_action)

            # --------------------------------------------------------
            # 2. 环境交互 (Env Step)
            # --------------------------------------------------------
            action_list = [a_fc, a_bat, a_sc]
            time_start_step = time.time()
            s_, r, done, _ = env.step(action_list)
            episode_times['Env_Step'] += (time.time() - time_start_step)

            # --------------------------------------------------------
            # 3. 存储转换 (Data Storage)
            # --------------------------------------------------------
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

            # --------------------------------------------------------
            # 4. I-DQN 独立学习 (DQN Learn)
            # --------------------------------------------------------
            if memory_counter[0] > MEMORY_CAPACITY and memory_counter[0] % LEARN_FREQUENCY == 0:
                time_start_learn = time.time()
                # 0 for FC action column, 1 for Bat action column, 2 for SC action column
                FC_Agent.learn(0)
                Bat_Agent.learn(1)
                SC_Agent.learn(2)
                episode_times['DQN_Learn'] += (time.time() - time_start_learn)

            if done:
                writer.add_scalar("Ep_r/Ep", ep_r, i_episode)
                using_time_total = time.time() - start_time_total

                # 使用 set_postfix 实时更新进度条信息
                episode_pbar.set_postfix({
                    'Ep_r': f'{ep_r:.2f}',
                    'Total_Time': f'{using_time_total:.2f}s',
                    'Env_Step_Time_ms': f"{(episode_times['Env_Step'] / step_count) * 1000:.2f}",
                })

                # ❗ 打印详细的耗时分解结果（仅在前 5 回合和每 10 回合打印）
                if i_episode < 2 or (i_episode + 1) % 100 == 0:
                    print_time_breakdown(i_episode + 1, episode_times)

                break

            s = s_

        x.append(int(i_episode))
        y.append(float(ep_r))

        # --------------------------------------------------------
        # 保存最优模型
        # --------------------------------------------------------
        if ep_r > reward_max:
            reward_max = ep_r
            net_name_base = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{i_episode + 1}"
                             f"_pool{POOL_SIZE}_freq{LEARN_FREQUENCY}_MARL_{remark}_MAX_R{int(reward_max)}")

            # 必须保存所有三个智能体模型
            torch.save(FC_Agent.eval_net.state_dict(), f"{net_name_base}_FC.pth")
            torch.save(Bat_Agent.eval_net.state_dict(), f"{net_name_base}_BAT.pth")
            torch.save(SC_Agent.eval_net.state_dict(), f"{net_name_base}_SC.pth")
            print(f"\nNew Max Value Models saved: {net_name_base}")  # 加换行防止被 tqdm 覆盖

            if REAL_TIME_DRAW:
                ax.plot(i_episode, reward_max, 'ro')

        # 实时绘图
        if REAL_TIME_DRAW:
            line.set_xdata(x)
            line.set_ydata(y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    # 最终模型保存
    final_net_name_base = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{EPISODE}_pool{POOL_SIZE}"
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

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    plt.savefig(f"{base_path}/train_curve_MARL_IQL_bs{BATCH_SIZE}_lr{int(LR * 10000)}_ep{EPISODE}.svg")

    if REAL_TIME_DRAW:
        plt.ioff()
    plt.show()