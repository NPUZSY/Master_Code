import numpy as np
import math
import torch
from Model.Battery import BatterySimple
from Model.FC import FCS
from Power_Profile import UAV_Load
import gym
from gym import spaces


class Envs(gym.Env):  # 继承自 gym.Env
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.ALPHA = 50000
        self.BETA = 7
        self.GAMMA = 1
        self.calorific_value = 143000  # 1g氢气能创造多少J能量
        self.temperature = UAV_Load.get_loads()[0]
        self.loads = UAV_Load.get_loads()[1]
        self.step_length = len(self.loads)
        self.battery = BatterySimple()
        self.fuel_cell = FCS()
        self.time_stamp = 0
        self.N_ACTIONS = 13

        # 使用 gym.spaces 定义动作空间和观测空间
        self.action_space = spaces.Discrete(self.N_ACTIONS)  # 离散动作空间
        # 定义观测空间的范围
        # [power_demand, temperature, self.power_fc, power_bat, soc_diff, soc_err]
        self.observation_space = spaces.Box(
            low=np.array([0, -100, 0, -3000, -1, -1]),
            high=np.array([8000, 20, 8000, 3000, 1, 1]),
            dtype=np.float32
        )

        self.power_fc = 0
        self.FC_MAX_POWER = 30000
        self.FC_MIN_POWER = 0
        self.current_observation = np.zeros(6)  # 用于存储当前观测值的变量

    def reset(self, **kwargs):
        self.time_stamp = 0
        self.battery = BatterySimple()
        self.power_fc = 0

        # state: power_demand, power_fc, power_bat, soc_diff, soc_err
        soc_diff, soc_err, _, power_bat, _ = self.battery.work(0)
        # 创建当前观测值，而不是修改 self.observation_space
        temperature = 0
        self.current_observation = np.array([0, self.power_fc, temperature, power_bat, soc_diff, soc_err], dtype=np.float32)
        self.power_fc = 0

        # 只返回观测值（符合 Gym API）
        return self.current_observation

    def step(self, action):
        # 从当前观测获取值
        if self.current_observation is None:
            # 如果没有当前观测值，使用默认值或重置环境
            self.reset()

        power_demand = self.current_observation[0]
        temperature = self.current_observation[1]
        self.power_fc = self.current_observation[2]
        power_bat = self.current_observation[3]
        soc_diff = self.current_observation[4]
        soc_err = self.current_observation[5]


        # 增量式action，将action动作叠加到当前的FC输出功率上
        self.power_fc += np.linspace(-300, 300, self.N_ACTIONS)[action]
        if self.power_fc > self.FC_MAX_POWER:
            self.power_fc = self.FC_MAX_POWER
        if self.power_fc < self.FC_MIN_POWER:
            self.power_fc = self.FC_MIN_POWER

        # 燃料电池输出功率乘当前功率的效率得到实际输出功率，剩余功率依靠锂电池补足
        power_bat = power_demand - self.power_fc
        # 由于每一秒计算一次，功率（瓦特）即为能量消耗（焦耳）
        soc_diff, soc_err, _, power_bat, _ = self.battery.work(power_bat)
        c_h2 = self.power_fc / self.calorific_value

        reward = -abs(5000 * soc_err)  # 扩大每次reward

        self.time_stamp += 1

        done = bool(self.time_stamp == len(self.loads) - 1)

        # 获取下一时刻的功率需求和温度
        if not done:
            power_demand = self.loads[self.time_stamp]
            temperature = self.temperature[self.time_stamp]
        else:
            power_demand = 0  # 终止状态的功率需求可以设为0
            temperature = 0  # 终止状态的温度需求可以设为0

        # 创建下一个观测值，而不是修改 self.observation_space
        self.current_observation = np.array([power_demand, temperature, self.power_fc, power_bat, soc_diff, soc_err],
                                            dtype=np.float32)

        # 返回四元组：(observation, reward, done, info)
        info_ = {}  # 可以包含额外信息
        return self.current_observation, reward, done, info_  # 返回当前观测值，奖励，是否终止，额外信息

    def render(self, mode='human'):
        # 可视化方法，可以为空
        pass

    def close(self):
        # 清理资源的方法，可以为空
        pass


if __name__ == '__main__':
    env = Envs()
    obs = env.reset()
    print("Reset observation:", obs)

    obs, reward, done, info = env.step(0)
    print("Step 0:")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")