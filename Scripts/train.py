import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from Env_FC_Li import Envs

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = Envs()
writer = SummaryWriter()
torch.set_default_dtype(torch.float32)
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.005  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
POOL_SIZE = 10  # memory pool size(episodes)
EPISODE = 1000
LEARN_FREQUENCY = 1

REAL_TIME_DRAW = False

MEMORY_CAPACITY = env.step_length * POOL_SIZE
current_timestamp = time.time()
local_time = time.localtime(current_timestamp)
execute_date = time.strftime("%m%d", local_time)
remark = "little_fc_power"



# env = env.unwrapped
N_ACTIONS = env.N_ACTIONS
N_STATES = env.observation_space.shape[0]

Base_Model_Name = ""


class Net(nn.Module):
    def __init__(self, ):
        torch.manual_seed(0)
        super(Net, self).__init__()
        # Input Layer
        self.input = nn.Linear(N_STATES, 20)
        self.input.weight.data.normal_(0, 0.1)  # initialization

        # Hidden Layer
        # self.lay1 = nn.Linear(20, 100)
        # self.lay1.weight.data.normal_(0, 0.1)   # initialization
        # self.lay2 = nn.Linear(100, 50)
        # self.lay2.weight.data.normal_(0, 0.1)  # initialization
        # self.lay3 = nn.Linear(50, 20)
        # self.lay3.weight.data.normal_(0, 0.1)  # initialization

        # Hidden Layer2
        self.lay_single = nn.Linear(20, 20)
        self.lay_single.weight.data.normal_(0, 0.1)  # initialization

        # Output Layer
        self.output = nn.Linear(20, N_ACTIONS)
        self.output.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.lay_single(x)
        x = F.relu(x)
        actions_value = self.output(x)
        return actions_value


class DQN(object):
    def __init__(self):

        self.env = env

        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def load_net(self, path):
        # self.eval_net, self.target_net = Net(), Net()
        self.eval_net.load_state_dict(torch.load(path))
        self.eval_net.to(device)
        # self.target_net.load_state_dict(torch.load(path))

    def choose_action(self, state_input: torch.Tensor, train=True):
        temp = torch.FloatTensor(state_input)
        state_input = torch.unsqueeze(temp.to(device), 0)
        if train:
            epsilon = 1
        else:
            epsilon = EPSILON
        # input only one sample
        if np.random.uniform() < epsilon:  # greedy
            actions_value = self.eval_net.forward(state_input)
            action_index = torch.max(actions_value, 1)[1].data
            action_index.to('cpu').numpy()
        else:  # random
            action_index = np.random.randint(0, N_ACTIONS)
        return action_index

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # print(transition)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("Start Learn...")
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # writer.add_scalar("Loss/Learn", loss, self.learn_step_counter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print("Learn Finished!")


def get_max_folder_name(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"指定的目录 {directory} 不存在。")
        return None
    # 获取目录下所有文件夹的名称
    folders = [int(name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    # 如果没有文件夹，返回 None
    if not folders:
        print(f"指定的目录 {directory} 下没有文件夹。")
        return 0
    # 找出最大的文件夹名
    max_folder = max(folders)
    return max_folder + 1


if __name__ == '__main__':
    os.makedirs(f"../nets/{execute_date}", exist_ok=True)
    train_id = get_max_folder_name(f"../nets/{execute_date}")
    base_path = f"../nets/{execute_date}/{train_id}"
    os.makedirs(base_path)
    net_name = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{EPISODE}_pool{POOL_SIZE}"
                f"_freq{LEARN_FREQUENCY}_basemodel_{Base_Model_Name}_{remark}")
    start_time = time.time()
    dqn = DQN()
    if Base_Model_Name:
        dqn.load_net(f"../nets/0227/27/{Base_Model_Name}")
    if REAL_TIME_DRAW:
        plt.ion()  # 打开交互式模式
    fig, ax = plt.subplots()

    x = []
    y = []
    line, = ax.plot(x, y)
    reward_max = -1e6
    print('\nCollecting experience...')
    for i_episode in range(EPISODE):
        s = env.reset()
        ep_r = 0

        while True:
            # env.render()
            a = dqn.choose_action(s).to('cpu')

            # take action
            s_, r, done, _ = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY and dqn.memory_counter % LEARN_FREQUENCY == 0:
                dqn.learn()

            if done:
                writer.add_scalar("Ep_r/Ep", ep_r, i_episode)
                using_time = time.time() - start_time
                print(f'Ep: {i_episode} | Ep_r: {ep_r} time:{using_time}s')
                break

            s = s_

        x.append(int(i_episode))
        y.append(float(ep_r))  # 曲线的示例数据，请根据您的需求更改
        if ep_r > reward_max:
            reward_max = ep_r
            net_name = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{i_episode + 1}"
                        f"_pool{POOL_SIZE}_freq{LEARN_FREQUENCY}_basemodel_{Base_Model_Name}_{remark}")
            torch.save(dqn.eval_net.state_dict(), net_name)
            print(f"New Max Value Model:{net_name}")
            ax.plot(i_episode, reward_max, 'ro')

        if REAL_TIME_DRAW:
            line.set_xdata(x)
            line.set_ydata(y)

            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # 暂停一段时间以实现实时更新

    net_name = (f"{base_path}/bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{EPISODE}_pool{POOL_SIZE}"
                f"_freq{LEARN_FREQUENCY}_basemodel_{Base_Model_Name}_{remark}")
    torch.save(dqn.eval_net.state_dict(), net_name)

    # model = dqn.eval_net
    # writer.add_graph(model)

    writer.flush()
    writer.close()
    print(f"net name:{net_name}")

    line.set_xdata(x)
    line.set_ydata(y)

    ax.relim()
    ax.autoscale_view()
    # 保存前最大化窗口
    plt.get_current_fig_manager().window.showMaximized()
    plt.savefig(f"{base_path}/train_curve_bs{BATCH_SIZE}_lr{int(LR * 10000)}_episode_{EPISODE}_pool{POOL_SIZE}"
                f"_freq{LEARN_FREQUENCY}_basemodel_{Base_Model_Name}_{remark}.svg")
    if REAL_TIME_DRAW:
        plt.ioff()
    plt.show()
