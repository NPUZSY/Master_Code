import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，确保在没有图形界面的环境中也能运行
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到Python路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入工具函数
from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# 设置字体为Times New Roman（确保图表字体规范）
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

from Model.Battery import BatterySimple
from Model.FC import FCS
from Model.SuperCap import SuperCapacitor


class EnvUltra(gym.Env):
    """
    超级环境：支持多种场景功率剖面的三源耦合环境（FC + Battery + SuperCap）
    - 支持三种场景：cruise（长航时巡航）、recon（跨域侦察）、rescue（应急救援）
    - 每种场景具有不同的功率和温度剖面
    - 完全兼容原 Env 的接口，可作为原 Env 的超集使用
    - 适配三智能体 I-DQN 架构的动作输入（现为动作列表）
    - 核心特性：
      1. 锂电池功率由智能体动作直接决定，超级电容补偿功率差值
      2. 超级电容满充/放空时继续充/放电，按剩余功率的10倍惩罚
      3. 支持通过 scenario_type 参数选择不同场景
    """

    # 经典环境场景类型（兼容旧智能体）
    CLASSIC_SCENARIO_TYPES = ['default']
    # 超级环境场景类型
    ADVANCED_SCENARIO_TYPES = ['cruise', 'recon', 'rescue',
                              'air', 'surface', 'underwater',
                              'air_to_surface', 'surface_to_air',
                              'air_to_underwater', 'underwater_to_air',
                              'surface_to_underwater', 'underwater_to_surface']
    # 合并所有场景类型
    SCENARIO_TYPES = CLASSIC_SCENARIO_TYPES + ADVANCED_SCENARIO_TYPES

    def __init__(self, scenario_type='cruise'):
        super().__init__()

        if scenario_type not in self.SCENARIO_TYPES:
            raise ValueError(f"Invalid scenario_type: {scenario_type}. Must be one of {self.SCENARIO_TYPES}")

        self.scenario_type = scenario_type

        self.dt = 1.0
        self.calorific_value = 143000.0
        self.Q_H2_eq = 142000.0

        self.P_FC_MAX = 5000.0
        self.P_FC_MIN = 0.0
        self.P_BAT_MAX = 5000.0
        self.P_SC_MAX = 2000.0

        self.w1 = -200
        self.w2 = -0.1
        self.w3 = -0.1
        self.w_sc_punish = 10
        self.minmatch_punish = 10
        self.w_fc_tracking = -0.1  # 燃料电池跟踪负载奖励项权重

        if self.w1 + self.w2 + self.w3 >= 0:
            print("警告：奖励权重之和非负，可能导致训练异常。")

        self._build_scenario_profiles()

        self.battery = BatterySimple()
        self.fuel_cell = FCS()
        self.supercap = SuperCapacitor()

        self.K_FC_MIN = -15
        self.K_FC_MAX = 16
        self.K_BAT_MIN = -20
        self.K_BAT_MAX = 19

        self.N_FC_ACTIONS = self.K_FC_MAX - self.K_FC_MIN + 1
        self.N_BAT_ACTIONS = self.K_BAT_MAX - self.K_BAT_MIN + 1
        self.N_SC_ACTIONS = 2
        self.N_ACTIONS = self.N_FC_ACTIONS * self.N_BAT_ACTIONS * self.N_SC_ACTIONS

        self.action_space = spaces.Dict({
            'fc': spaces.Discrete(self.N_FC_ACTIONS),
            'bat': spaces.Discrete(self.N_BAT_ACTIONS),
            'sc': spaces.Discrete(self.N_SC_ACTIONS)
        })

        self.observation_space = spaces.Box(
            low=np.array([0., -100., 0., -self.P_BAT_MAX, -self.P_SC_MAX, 0., 0.], dtype=np.float32),
            high=np.array([80000., 200., self.P_FC_MAX, self.P_BAT_MAX, self.P_SC_MAX, 1., 1.], dtype=np.float32),
            dtype=np.float32
        )

        self.time_stamp = 0
        self.power_fc = 0.0
        self.r_fc_accum = 0.0
        self.punish_step = 1.0
        self.punish_decay = 0.5
        self.r_sc_punish = 0.0
        self.reset()

    def _build_scenario_profiles(self):
        """构建场景功率和温度剖面"""
        # 处理经典环境的默认场景
        if self.scenario_type == 'default':
            # 使用经典环境的功率剖面生成方式
            try:
                from Scripts.Power_Profile import UAV_Load
                loads_data = UAV_Load.get_loads()
                self.temperature = loads_data[0]
                self.power_profile = loads_data[1]
                self.loads = self.power_profile  # 确保设置self.loads属性
                self.mode_annotations = [{'type': 'surface', 'start': 0, 'end': len(self.power_profile)}]
                self.step_length = len(self.power_profile)
                return
            except Exception as e:
                # 如果导入失败，使用默认值
                print(f"警告: 无法加载 UAV_Load, 使用默认值。错误: {e}")
                self.TOTAL_DURATION = 600
                self.temperature = np.array([25.0] * self.TOTAL_DURATION)
                self.power_profile = np.array([1000.0] * self.TOTAL_DURATION)
                self.loads = self.power_profile  # 确保设置self.loads属性
                self.mode_annotations = [{'type': 'surface', 'start': 0, 'end': self.TOTAL_DURATION}]
                self.step_length = self.TOTAL_DURATION
                return
        
        # 超级环境的场景处理
        self.TOTAL_DURATION = 1800
        self.SWITCH_DURATION = 50
        self.T_AIR = 0
        self.T_SURFACE = 20
        self.T_UNDERWATER = 5
        self.P_AIR_BASE = 2500
        self.P_SURFACE_BASE = 1000
        self.P_UNDERWATER_BASE = 3000

        self.POWER_FLUCTUATION_RATIO_CRUISE = 0.02
        self.POWER_FLUCTUATION_RATIO_RECON_AIR = 0.02
        self.POWER_FLUCTUATION_RATIO_RECON_SURFACE = 0.02
        self.POWER_FLUCTUATION_RATIO_RECON_UNDERWATER = 0.10
        self.POWER_FLUCTUATION_RATIO_RESCUE = 0.05
        self.POWER_FLUCTUATION_RATIO_SINGLE_MODE = 0.05
        self.POWER_FLUCTUATION_MAX = 500

        time_points = np.arange(0, self.TOTAL_DURATION, 1)
        power_profile = np.zeros_like(time_points, dtype=float)
        mode_annotations = []

        if self.scenario_type == 'cruise':
            air1_time = time_points[0:600]
            air1_power = self._generate_bayesian_power(air1_time, self.P_AIR_BASE, 'cruise', 'air')
            power_profile[0:600] = air1_power
            mode_annotations.append({'type': 'air', 'start': 0, 'end': 600})

            switch1_time = time_points[600:650]
            switch1_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_AIR_BASE, self.P_SURFACE_BASE, 'air_to_surface')
            power_profile[600:650] = switch1_power
            mode_annotations.append({'type': 'air_to_surface_switch', 'start': 600, 'end': 650})

            surface_time = time_points[650:1150]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'cruise', 'surface')
            power_profile[650:1150] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 650, 'end': 1150})

            switch2_time = time_points[1150:1200]
            switch2_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_AIR_BASE, 'surface_to_air')
            power_profile[1150:1200] = switch2_power
            mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1150, 'end': 1200})

            air2_time = time_points[1200:1800]
            air2_power = self._generate_bayesian_power(air2_time, self.P_AIR_BASE, 'cruise', 'air')
            power_profile[1200:1800] = air2_power
            mode_annotations.append({'type': 'air', 'start': 1200, 'end': 1800})

        elif self.scenario_type == 'recon':
            air1_time = time_points[0:200]
            air1_power = self._generate_bayesian_power(air1_time, self.P_AIR_BASE, 'recon', 'air')
            power_profile[0:200] = air1_power
            mode_annotations.append({'type': 'air', 'start': 0, 'end': 200})

            switch1_time = time_points[200:250]
            switch1_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_AIR_BASE, self.P_UNDERWATER_BASE, 'air_to_underwater')
            power_profile[200:250] = switch1_power
            mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 200, 'end': 250})

            underwater_time = time_points[250:1300]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'recon', 'underwater')
            power_profile[250:1300] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 250, 'end': 1300})

            switch2_time = time_points[1300:1350]
            switch2_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_UNDERWATER_BASE, self.P_SURFACE_BASE, 'underwater_to_surface')
            power_profile[1300:1350] = switch2_power
            mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1300, 'end': 1350})

            surface_time = time_points[1350:1550]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'recon', 'surface')
            power_profile[1350:1550] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 1350, 'end': 1550})

            switch3_time = time_points[1550:1600]
            switch3_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_AIR_BASE, 'surface_to_air')
            power_profile[1550:1600] = switch3_power
            mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1550, 'end': 1600})

            air2_time = time_points[1600:1800]
            air2_power = self._generate_bayesian_power(air2_time, self.P_AIR_BASE, 'recon', 'air')
            power_profile[1600:1800] = air2_power
            mode_annotations.append({'type': 'air', 'start': 1600, 'end': 1800})

        elif self.scenario_type == 'rescue':
            surface1_time = time_points[0:320]
            surface1_power = self._generate_bayesian_power(surface1_time, self.P_SURFACE_BASE, 'rescue', 'surface')
            power_profile[0:320] = surface1_power
            mode_annotations.append({'type': 'surface', 'start': 0, 'end': 320})

            switch1_time = time_points[320:370]
            switch1_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_AIR_BASE, 'surface_to_air')
            power_profile[320:370] = switch1_power
            mode_annotations.append({'type': 'surface_to_air_switch', 'start': 320, 'end': 370})

            air1_time = time_points[370:690]
            air1_power = self._generate_bayesian_power(air1_time, self.P_AIR_BASE, 'rescue', 'air')
            power_profile[370:690] = air1_power
            mode_annotations.append({'type': 'air', 'start': 370, 'end': 690})

            switch2_time = time_points[690:740]
            switch2_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_AIR_BASE, self.P_UNDERWATER_BASE, 'air_to_underwater')
            power_profile[690:740] = switch2_power
            mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 690, 'end': 740})

            underwater_time = time_points[740:1060]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'rescue', 'underwater')
            power_profile[740:1060] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 740, 'end': 1060})

            switch3_time = time_points[1060:1110]
            switch3_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_UNDERWATER_BASE, self.P_SURFACE_BASE, 'underwater_to_surface')
            power_profile[1060:1110] = switch3_power
            mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1060, 'end': 1110})

            surface2_time = time_points[1110:1430]
            surface2_power = self._generate_bayesian_power(surface2_time, self.P_SURFACE_BASE, 'rescue', 'surface')
            power_profile[1110:1430] = surface2_power
            mode_annotations.append({'type': 'surface', 'start': 1110, 'end': 1430})

            switch4_time = time_points[1430:1480]
            switch4_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_AIR_BASE, 'surface_to_air')
            power_profile[1430:1480] = switch4_power
            mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1430, 'end': 1480})

            air2_time = time_points[1480:1800]
            air2_power = self._generate_bayesian_power(air2_time, self.P_AIR_BASE, 'rescue', 'air')
            power_profile[1480:1800] = air2_power
            mode_annotations.append({'type': 'air', 'start': 1480, 'end': 1800})
        
        # 新增：单模态场景
        elif self.scenario_type == 'air':
            # 全程空中模式
            air_time = time_points[:]
            air_power = self._generate_bayesian_power(air_time, self.P_AIR_BASE, 'single_mode', 'air')
            power_profile[:] = air_power
            mode_annotations.append({'type': 'air', 'start': 0, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'surface':
            # 全程水面模式
            surface_time = time_points[:]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'single_mode', 'surface')
            power_profile[:] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 0, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'underwater':
            # 全程水下模式
            underwater_time = time_points[:]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'single_mode', 'underwater')
            power_profile[:] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 0, 'end': self.TOTAL_DURATION})
        
        # 新增：切换模态场景
        elif self.scenario_type == 'air_to_surface':
            # 空中到水面切换
            air_time = time_points[0:875]
            air_power = self._generate_bayesian_power(air_time, self.P_AIR_BASE, 'single_mode', 'air')
            power_profile[0:875] = air_power
            mode_annotations.append({'type': 'air', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_AIR_BASE, self.P_SURFACE_BASE, 'air_to_surface')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'air_to_surface_switch', 'start': 875, 'end': 925})
            
            surface_time = time_points[925:]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'single_mode', 'surface')
            power_profile[925:] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 925, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'surface_to_air':
            # 水面到空中切换
            surface_time = time_points[0:875]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'single_mode', 'surface')
            power_profile[0:875] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_AIR_BASE, 'surface_to_air')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'surface_to_air_switch', 'start': 875, 'end': 925})
            
            air_time = time_points[925:]
            air_power = self._generate_bayesian_power(air_time, self.P_AIR_BASE, 'single_mode', 'air')
            power_profile[925:] = air_power
            mode_annotations.append({'type': 'air', 'start': 925, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'air_to_underwater':
            # 空中到水下切换
            air_time = time_points[0:875]
            air_power = self._generate_bayesian_power(air_time, self.P_AIR_BASE, 'single_mode', 'air')
            power_profile[0:875] = air_power
            mode_annotations.append({'type': 'air', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_AIR_BASE, self.P_UNDERWATER_BASE, 'air_to_underwater')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 875, 'end': 925})
            
            underwater_time = time_points[925:]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'single_mode', 'underwater')
            power_profile[925:] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 925, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'underwater_to_air':
            # 水下到空中切换
            underwater_time = time_points[0:875]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'single_mode', 'underwater')
            power_profile[0:875] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_UNDERWATER_BASE, self.P_AIR_BASE, 'underwater_to_air')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'underwater_to_air_switch', 'start': 875, 'end': 925})
            
            air_time = time_points[925:]
            air_power = self._generate_bayesian_power(air_time, self.P_AIR_BASE, 'single_mode', 'air')
            power_profile[925:] = air_power
            mode_annotations.append({'type': 'air', 'start': 925, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'surface_to_underwater':
            # 水面到水下切换
            surface_time = time_points[0:875]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'single_mode', 'surface')
            power_profile[0:875] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_SURFACE_BASE, self.P_UNDERWATER_BASE, 'surface_to_underwater')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'surface_to_underwater_switch', 'start': 875, 'end': 925})
            
            underwater_time = time_points[925:]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'single_mode', 'underwater')
            power_profile[925:] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 925, 'end': self.TOTAL_DURATION})
        
        elif self.scenario_type == 'underwater_to_surface':
            # 水下到水面切换
            underwater_time = time_points[0:875]
            underwater_power = self._generate_bayesian_power(underwater_time, self.P_UNDERWATER_BASE, 'single_mode', 'underwater')
            power_profile[0:875] = underwater_power
            mode_annotations.append({'type': 'underwater', 'start': 0, 'end': 875})
            
            switch_time = time_points[875:925]
            switch_power = self._generate_switch_power(np.linspace(0, 50, 50), self.P_UNDERWATER_BASE, self.P_SURFACE_BASE, 'underwater_to_surface')
            power_profile[875:925] = switch_power
            mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 875, 'end': 925})
            
            surface_time = time_points[925:]
            surface_power = self._generate_bayesian_power(surface_time, self.P_SURFACE_BASE, 'single_mode', 'surface')
            power_profile[925:] = surface_power
            mode_annotations.append({'type': 'surface', 'start': 925, 'end': self.TOTAL_DURATION})

        self.time_points = time_points
        self.power_profile = power_profile
        self.temperature = self._generate_temperature_curve(time_points, mode_annotations, self.scenario_type)
        self.mode_annotations = mode_annotations
        self.loads = power_profile
        self.step_length = len(self.loads)

    def _generate_bayesian_power(self, time_points, base_power, scenario_type, mode_type):
        """基于贝叶斯模型生成带随机性的功率曲线"""
        if scenario_type == 'cruise':
            std = min(base_power * self.POWER_FLUCTUATION_RATIO_CRUISE, self.POWER_FLUCTUATION_MAX)
        elif scenario_type == 'recon':
            if mode_type == 'underwater':
                std = min(base_power * self.POWER_FLUCTUATION_RATIO_RECON_UNDERWATER, self.POWER_FLUCTUATION_MAX)
            elif mode_type == 'air':
                std = min(base_power * self.POWER_FLUCTUATION_RATIO_RECON_AIR, self.POWER_FLUCTUATION_MAX)
            else:
                std = min(base_power * self.POWER_FLUCTUATION_RATIO_RECON_SURFACE, self.POWER_FLUCTUATION_MAX)
        elif scenario_type == 'rescue':
            std = min(base_power * self.POWER_FLUCTUATION_RATIO_RESCUE, self.POWER_FLUCTUATION_MAX)
        elif scenario_type == 'single_mode':
            std = min(base_power * self.POWER_FLUCTUATION_RATIO_SINGLE_MODE, self.POWER_FLUCTUATION_MAX)
        else:
            std = min(base_power * 0.05, self.POWER_FLUCTUATION_MAX)

        np.random.seed(42)
        power_fluctuation = norm.rvs(loc=0, scale=std, size=len(time_points))
        power = base_power + power_fluctuation

        if scenario_type == 'rescue':
            peak_factor = 1.3
            peak_indices = np.random.choice(len(time_points), size=int(len(time_points)*0.1), replace=False)
            power[peak_indices] = power[peak_indices] * peak_factor

        power = np.maximum(power, 0)
        return power

    def _generate_switch_power(self, time_points, start_power, end_power, switch_type):
        """生成切换模态的功率曲线"""
        power = np.zeros_like(time_points, dtype=np.float64)
        for i, ti in enumerate(time_points):
            if ti <= 10:
                power[i] = start_power
            elif ti <= 20:
                ratio = (ti - 10) / 10
                if switch_type == 'air_to_surface':
                    power[i] = start_power + ratio * (1.1 * start_power - start_power)
                elif switch_type == 'surface_to_air':
                    power[i] = start_power + ratio * (0.9 * start_power - start_power)
                elif switch_type == 'air_to_underwater':
                    power[i] = start_power
                elif switch_type == 'underwater_to_air':
                    power[i] = start_power + ratio * (1.1 * start_power - start_power)
                elif switch_type == 'surface_to_underwater':
                    power[i] = start_power + ratio * (1.05 * start_power - start_power)
                elif switch_type == 'underwater_to_surface':
                    power[i] = start_power + ratio * (0.95 * start_power - start_power)
            elif ti <= 35:
                ratio = (ti - 20) / 15
                if switch_type == 'air_to_surface':
                    power[i] = 1.1 * start_power + ratio * (0.9 * end_power - 1.1 * start_power)
                elif switch_type == 'surface_to_air':
                    power[i] = 0.9 * start_power + ratio * (2.0 * end_power - 0.9 * start_power)
                elif switch_type == 'air_to_underwater':
                    power[i] = start_power
                elif switch_type == 'underwater_to_air':
                    power[i] = 1.1 * start_power + ratio * (2.0 * end_power - 1.1 * start_power)
                elif switch_type == 'surface_to_underwater':
                    power[i] = 1.05 * start_power + ratio * (1.1 * end_power - 1.05 * start_power)
                elif switch_type == 'underwater_to_surface':
                    power[i] = 0.95 * start_power + ratio * (0.9 * end_power - 0.95 * start_power)
            elif ti <= 40:
                ratio = (ti - 35) / 5
                if switch_type == 'air_to_surface':
                    power[i] = 0.9 * end_power + ratio * (end_power - 0.9 * end_power)
                elif switch_type == 'surface_to_air':
                    power[i] = 2.0 * end_power + ratio * (end_power - 2.0 * end_power)
                elif switch_type == 'air_to_underwater':
                    power[i] = end_power
                elif switch_type == 'underwater_to_air':
                    power[i] = 2.0 * end_power + ratio * (end_power - 2.0 * end_power)
                elif switch_type == 'surface_to_underwater':
                    power[i] = 1.1 * end_power + ratio * (end_power - 1.1 * end_power)
                elif switch_type == 'underwater_to_surface':
                    power[i] = 0.9 * end_power + ratio * (end_power - 0.9 * end_power)
            else:
                power[i] = end_power
        power = np.maximum(power, 0)
        return power

    def _generate_temperature_curve(self, time_points, mode_sequence, scenario_type):
        """生成温度变化曲线"""
        temp = np.zeros_like(time_points, dtype=np.float64)
        time_idx = 0
        rate_factor = 1.5 if scenario_type == 'rescue' else 1.0

        for segment in mode_sequence:
            seg_type, seg_start, seg_end = segment['type'], segment['start'], segment['end']
            seg_len = seg_end - seg_start
            seg_time = time_points[time_idx:time_idx+seg_len]

            if seg_type == 'air':
                seg_temp = np.full(seg_len, self.T_AIR, dtype=np.float64)
            elif seg_type == 'surface':
                seg_temp = np.full(seg_len, self.T_SURFACE, dtype=np.float64)
            elif seg_type == 'underwater':
                seg_temp = np.full(seg_len, self.T_UNDERWATER, dtype=np.float64)
            elif 'switch' in seg_type:
                seg_temp = np.zeros(seg_len, dtype=np.float64)
                switch_time = np.linspace(0, self.SWITCH_DURATION, seg_len)

                for i, ti in enumerate(switch_time):
                    if ti <= 10:
                        if seg_type == 'air_to_surface_switch':
                            seg_temp[i] = self.T_AIR
                        elif seg_type == 'surface_to_air_switch':
                            seg_temp[i] = self.T_SURFACE
                        elif seg_type == 'air_to_underwater_switch':
                            seg_temp[i] = self.T_AIR
                        elif seg_type == 'underwater_to_surface_switch':
                            seg_temp[i] = self.T_UNDERWATER
                        elif seg_type == 'surface_to_underwater_switch':
                            seg_temp[i] = self.T_SURFACE
                        elif seg_type == 'underwater_to_air_switch':
                            seg_temp[i] = self.T_UNDERWATER
                    elif 10 < ti <= 40:
                        ratio = ((ti - 10) / 30) * rate_factor
                        ratio = np.clip(ratio, 0, 1)

                        if seg_type == 'air_to_surface_switch':
                            seg_temp[i] = self.T_AIR + ratio * (self.T_SURFACE - self.T_AIR)
                        elif seg_type == 'surface_to_air_switch':
                            seg_temp[i] = self.T_SURFACE - ratio * (self.T_SURFACE - self.T_AIR)
                        elif seg_type == 'air_to_underwater_switch':
                            if ti <= 25:
                                ratio_sub = ((ti - 10) / 15) * rate_factor
                                seg_temp[i] = self.T_AIR + ratio_sub * (self.T_SURFACE - self.T_AIR)
                            else:
                                ratio_sub = ((ti - 25) / 15) * rate_factor
                                seg_temp[i] = self.T_SURFACE - ratio_sub * (self.T_SURFACE - self.T_UNDERWATER)
                        elif seg_type == 'underwater_to_surface_switch':
                            seg_temp[i] = self.T_UNDERWATER + ratio * (self.T_SURFACE - self.T_UNDERWATER)
                        elif seg_type == 'surface_to_underwater_switch':
                            seg_temp[i] = self.T_SURFACE - ratio * (self.T_SURFACE - self.T_UNDERWATER)
                        elif seg_type == 'underwater_to_air_switch':
                            if ti <= 25:
                                ratio_sub = ((ti - 10) / 15) * rate_factor
                                seg_temp[i] = self.T_UNDERWATER + ratio_sub * (self.T_SURFACE - self.T_UNDERWATER)
                            else:
                                ratio_sub = ((ti - 25) / 15) * rate_factor
                                seg_temp[i] = self.T_SURFACE - ratio_sub * (self.T_SURFACE - self.T_AIR)
                    else:
                        if seg_type == 'air_to_surface_switch':
                            seg_temp[i] = self.T_SURFACE
                        elif seg_type == 'surface_to_air_switch':
                            seg_temp[i] = self.T_AIR
                        elif seg_type == 'air_to_underwater_switch':
                            seg_temp[i] = self.T_UNDERWATER
                        elif seg_type == 'underwater_to_surface_switch':
                            seg_temp[i] = self.T_SURFACE
                        elif seg_type == 'surface_to_underwater_switch':
                            seg_temp[i] = self.T_UNDERWATER
                        elif seg_type == 'underwater_to_air_switch':
                            seg_temp[i] = self.T_AIR
            else:
                seg_temp = np.zeros(seg_len, dtype=np.float64)

            np.random.seed(42)
            temp_fluctuation = norm.rvs(loc=0, scale=0.5*rate_factor, size=seg_len)
            seg_temp += temp_fluctuation
            seg_temp = np.clip(seg_temp, -5, 25)

            temp[time_idx:time_idx+seg_len] = seg_temp
            time_idx += seg_len

        return temp

    def _fc_delta_from_index(self, idx):
        k = self.K_FC_MIN + int(idx)
        delta = k * 0.001 * self.P_FC_MAX
        return float(delta)

    def _bat_power_from_index(self, idx):
        k = self.K_BAT_MIN + int(idx)
        p = k * 0.05 * self.P_BAT_MAX
        return float(p)

    def reset(self, **kwargs):
        self.time_stamp = 0
        self.battery = BatterySimple()
        self.fuel_cell = FCS()
        self.supercap = SuperCapacitor()
        self.power_fc = 0.0
        self.r_fc_accum = 0.0
        self.r_sc_punish = 0.0

        P_load = float(self.loads[0])
        T_env = float(self.temperature[0]) if len(self.temperature) > 0 else 0.0

        try:
            soc_b = float(self.battery.soc)
        except Exception:
            soc_b = 0.5
        try:
            soc_sc = float(self.supercap.soc)
        except Exception:
            soc_sc = 0.5

        self.current_observation = np.array([P_load, T_env, self.power_fc, 0.0, 0.0, soc_b, soc_sc], dtype=np.float32)
        return self.current_observation

    def step(self, action_list):
        a_fc = int(action_list[0])
        a_bat = int(action_list[1])
        a_sc = int(action_list[2])

        action_decoded = {
            'fc': a_fc,
            'bat': a_bat,
            'sc': a_sc
        }

        P_load = float(self.current_observation[0])
        T_env = float(self.current_observation[1])

        delta_P_fc = self._fc_delta_from_index(action_decoded['fc'])
        P_bat_cmd = self._bat_power_from_index(action_decoded['bat'])
        sc_on = bool(int(action_decoded['sc']) == 1)

        self.power_fc = float(np.clip(self.power_fc + delta_P_fc, self.P_FC_MIN, self.P_FC_MAX))
        P_bat_final = float(np.clip(P_bat_cmd, -self.P_BAT_MAX, self.P_BAT_MAX))

        power_diff = P_load - self.power_fc - P_bat_final

        if sc_on:
            P_sc = float(np.clip(power_diff, -self.P_SC_MAX, self.P_SC_MAX))
        else:
            P_sc = 0.0

        try:
            work_ret = self.battery.work(P_bat_final)
            if isinstance(work_ret, tuple) or isinstance(work_ret, list):
                if len(work_ret) >= 3:
                    soc_diff, soc_err, actual_bat_power = work_ret[0], work_ret[1], work_ret[2]
                else:
                    soc_diff = work_ret[0]
                    soc_err = work_ret[1] if len(work_ret) > 1 else 0.0
                    actual_bat_power = P_bat_final
            else:
                soc_diff, soc_err, actual_bat_power = 0.0, 0.0, P_bat_final
        except Exception:
            try:
                soc_prev = float(self.battery.soc)
                energy_delta = P_bat_final * self.dt
                cap_total = getattr(self.battery, "capacity_total", getattr(self.battery, "capacity", 1.0))
                soc_new = max(0.0, min(1.0, soc_prev - energy_delta / (cap_total + 1e-9)))
                soc_diff = soc_prev - soc_new
                soc_err = soc_new - getattr(self.battery, "soc_ref", 0.6)
                self.battery.soc = soc_new
                actual_bat_power = P_bat_final
            except Exception:
                soc_diff, soc_err, actual_bat_power = 0.0, 0.0, P_bat_final

        try:
            i_sc, v_sc, soc_sc, actual_p_sc = self.supercap.output(P_sc)
        except Exception:
            actual_p_sc = P_sc
            try:
                if hasattr(self.supercap, 'soc'):
                    soc_sc = self.supercap.soc
                else:
                    soc_sc = 0.5
            except Exception:
                soc_sc = 0.5

        current_sc_punish = 0.0
        soc_sc_clamped = np.clip(soc_sc, 0.0, 1.0)
        if sc_on:
            if np.isclose(soc_sc_clamped, 1.0) and P_sc < 0:
                current_sc_punish = abs(P_sc) * self.w_sc_punish
            elif np.isclose(soc_sc_clamped, 0.0) and P_sc > 0:
                current_sc_punish = abs(P_sc) * self.w_sc_punish
        self.r_sc_punish += current_sc_punish

        P_fc = float(self.power_fc)
        eta_fc = None
        try:
            if hasattr(self.fuel_cell, "Eng_fuel_func"):
                try:
                    eta_fc = float(self.fuel_cell.Eng_fuel_func(P_fc / 1000.0))
                except Exception:
                    eta_fc = float(self.fuel_cell.Eng_fuel_func(P_fc))
            elif hasattr(self.fuel_cell, "cal_efficiency"):
                eta_fc = float(self.fuel_cell.cal_efficiency(P_fc))
        except Exception:
            eta_fc = None

        if eta_fc is None or math.isnan(eta_fc) or eta_fc <= 0:
            eta_fc = 0.45

        eta_conv = 0.95

        C_fc = 0.0
        C_bat = 0.0
        if P_fc > 0:
            C_fc = (P_fc * self.dt) / (max(1e-6, eta_fc * eta_conv) * self.calorific_value)
        C_bat = (actual_bat_power * self.dt) / (eta_conv * self.Q_H2_eq)

        if P_fc > 0.9 * self.P_FC_MAX:
            self.r_fc_accum += self.punish_step
        else:
            self.r_fc_accum = max(0.0, self.r_fc_accum - self.punish_decay)

        r_fc = float(self.r_fc_accum)

        try:
            soc_b = float(self.battery.soc)
        except Exception:
            soc_b = 0.5

        if soc_b < 0.2 or soc_b > 0.8:
            r_bat = 1.0
        else:
            r_bat = 0.0

        r_bat += abs(soc_b - 0.6) * 5

        power_loss = abs(P_load - self.power_fc - actual_bat_power - actual_p_sc)
        r_match = current_sc_punish + power_loss * self.minmatch_punish
        
        # 计算燃料电池跟踪负载的奖励项
        # fc_tracking_error = abs(P_load - self.power_fc)
        # r_fc_tracking = fc_tracking_error * self.w_fc_tracking

        reward = float(
            self.w1 * (C_fc + C_bat) +
            self.w2 * (r_fc + r_bat) +
            self.w3 * r_match
        ) / self.step_length * 10

        self.time_stamp += 1
        done = bool(self.time_stamp >= len(self.loads) - 1)

        if not done:
            next_load = float(self.loads[self.time_stamp])
            next_temp = float(self.temperature[self.time_stamp]) if len(self.temperature) > self.time_stamp else 0.0
        else:
            next_load = 0.0
            next_temp = 0.0

        self.current_observation = np.array([
            next_load,
            next_temp,
            self.power_fc,
            actual_bat_power,
            actual_p_sc,
            soc_b,
            soc_sc
        ], dtype=np.float32)

        info = {
            "P_load": P_load,
            "P_fc": P_fc,
            "P_bat": actual_bat_power,
            "P_sc": actual_p_sc,
            "C_fc_g": C_fc,
            "C_bat_g": C_bat,
            "r_fc": r_fc,
            "r_bat": r_bat,
            "r_match": r_match,
            "eta_fc": eta_fc,
            "T_amb": T_env,
            "power_diff": power_diff,
            "soc_sc": soc_sc_clamped,
            "current_sc_punish": current_sc_punish,
            "total_sc_punish": self.r_sc_punish,
            "scenario_type": self.scenario_type
        }

        return self.current_observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    @staticmethod
    def plot_switch_mode_power():
        """
        绘制Fig5-6：不同模式切换的功率和温度曲线
        对应原Fig5-6_Switch_Mode_Plot.py的功能
        """
        # 定义基础功率值（W）
        P_air = 2500
        P_water_surface = 1000
        P_underwater = 3000
        
        # 定义温度常量（℃）
        T_air = 0
        T_surface = 20
        T_underwater = 5
        
        # 定义各切换模态的功率切换函数
        def air_to_surface(t, P1=P_air, P2=P_water_surface):
            """空中-水面切换模态：P1→1.1P1→0.9P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 20:
                    ratio = (ti - 10) / 10
                    power[i] = P1 + ratio * (1.1 * P1 - P1)
                elif ti <= 35:
                    ratio = (ti - 20) / 15
                    power[i] = 1.1 * P1 + ratio * (0.9 * P2 - 1.1 * P1)
                elif ti <= 40:
                    ratio = (ti - 35) / 5
                    power[i] = 0.9 * P2 + ratio * (P2 - 0.9 * P2)
                else:
                    power[i] = P2
            return np.maximum(power, 0)
        
        def surface_to_air(t, P1=P_water_surface, P2=P_air):
            """水面-空中切换模态：P1→0.9P1→2.0P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 20:
                    ratio = (ti - 10) / 10
                    power[i] = P1 + ratio * (0.9 * P1 - P1)
                elif ti <= 35:
                    ratio = (ti - 20) / 15
                    power[i] = 0.9 * P1 + ratio * (2.0 * P2 - 0.9 * P1)
                elif ti <= 40:
                    ratio = (ti - 35) / 5
                    power[i] = 2.0 * P2 + ratio * (P2 - 2.0 * P2)
                else:
                    power[i] = P2
            return np.maximum(power, 0)
        
        def air_to_underwater(t, P1=P_air, P2=P_underwater):
            """空中-水下切换模态：P1→P1→1.2P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 25:
                    power[i] = P1
                elif ti <= 30:
                    ratio = (ti - 25) / 5
                    power[i] = P1 + ratio * (1.2 * P2 - P1)
                elif ti <= 40:
                    ratio = (ti - 30) / 10
                    power[i] = 1.2 * P2 + ratio * (P2 - 1.2 * P2)
                else:
                    power[i] = P2
            return np.maximum(power, 0)
        
        def underwater_to_air(t, P1=P_underwater, P2=P_air):
            """水下-空中切换模态：P1→1.1P1→2.0P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 20:
                    ratio = (ti - 10) / 10
                    power[i] = P1 + ratio * (1.1 * P1 - P1)
                elif ti <= 35:
                    ratio = (ti - 20) / 15
                    power[i] = 1.1 * P1 + ratio * (2.0 * P2 - 1.1 * P1)
                elif ti <= 40:
                    ratio = (ti - 35) / 5
                    power[i] = 2.0 * P2 + ratio * (P2 - 2.0 * P2)
                else:
                    power[i] = P2
            return np.maximum(power, 0)
        
        def surface_to_underwater(t, P1=P_water_surface, P2=P_underwater):
            """水面-水下切换模态：P1→1.05P1→1.1P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 25:
                    ratio = (ti - 10) / 15
                    power[i] = P1 + ratio * (1.05 * P1 - P1)
                elif ti <= 40:
                    ratio = (ti - 25) / 15
                    power[i] = 1.05 * P1 + ratio * (1.1 * P2 - 1.05 * P1)
                else:
                    power[i] = 1.1 * P2 + ((ti - 40)/10) * (P2 - 1.1 * P2)
            return np.maximum(power, 0)
        
        def underwater_to_surface(t, P1=P_underwater, P2=P_water_surface):
            """水下-水面切换模态：P1→0.95P1→0.9P2→P2"""
            power = np.zeros_like(t, dtype=np.float64)
            for i, ti in enumerate(t):
                if ti <= 10:
                    power[i] = P1
                elif ti <= 25:
                    ratio = (ti - 10) / 15
                    power[i] = P1 + ratio * (0.95 * P1 - P1)
                elif ti <= 35:
                    ratio = (ti - 25) / 10
                    power[i] = 0.95 * P1 + ratio * (0.9 * P2 - 0.95 * P1)
                elif ti <= 40:
                    ratio = (ti - 35) / 5
                    power[i] = 0.9 * P2 + ratio * (P2 - 0.9 * P2)
                else:
                    power[i] = P2
            return np.maximum(power, 0)
        
        # 定义各模态的温度变化函数
        def get_temperature_curve(mode_name, time):
            """生成对应模态的温度变化曲线"""
            temp = np.zeros_like(time, dtype=np.float64)
            for i, ti in enumerate(time):
                if ti <= 10:
                    # 前10s：维持初始温度
                    if mode_name == 'Air to Surface':
                        temp[i] = T_air
                    elif mode_name == 'Surface to Air':
                        temp[i] = T_surface
                    elif mode_name == 'Air to Underwater':
                        temp[i] = T_air
                    elif mode_name == 'Underwater to Air':
                        temp[i] = T_underwater
                    elif mode_name == 'Surface to Underwater':
                        temp[i] = T_surface
                    elif mode_name == 'Underwater to Surface':
                        temp[i] = T_underwater
                
                elif 10 < ti <= 40:
                    # 过渡阶段（10-40s）：线性变化
                    if mode_name == 'Air to Surface':
                        ratio = (ti - 10) / 30
                        temp[i] = T_air + ratio * (T_surface - T_air)
                    elif mode_name == 'Surface to Air':
                        ratio = (ti - 10) / 30
                        temp[i] = T_surface - ratio * (T_surface - T_air)
                    elif mode_name == 'Air to Underwater':
                        if ti <= 25:
                            ratio = (ti - 10) / 15
                            temp[i] = T_air + ratio * (T_surface - T_air)
                        else:
                            ratio = (ti - 25) / 15
                            temp[i] = T_surface - ratio * (T_surface - T_underwater)
                    elif mode_name == 'Underwater to Air':
                        if ti <= 25:
                            ratio = (ti - 10) / 15
                            temp[i] = T_underwater + ratio * (T_surface - T_underwater)
                        else:
                            ratio = (ti - 25) / 15
                            temp[i] = T_surface - ratio * (T_surface - T_air)
                    elif mode_name == 'Surface to Underwater':
                        ratio = (ti - 10) / 30
                        temp[i] = T_surface - ratio * (T_surface - T_underwater)
                    elif mode_name == 'Underwater to Surface':
                        ratio = (ti - 10) / 30
                        temp[i] = T_underwater + ratio * (T_surface - T_underwater)
                else:
                    # 后10s（40-50s）：维持目标温度
                    if mode_name == 'Air to Surface':
                        temp[i] = T_surface
                    elif mode_name == 'Surface to Air':
                        temp[i] = T_air
                    elif mode_name == 'Air to Underwater':
                        temp[i] = T_underwater
                    elif mode_name == 'Underwater to Air':
                        temp[i] = T_air
                    elif mode_name == 'Surface to Underwater':
                        temp[i] = T_underwater
                    elif mode_name == 'Underwater to Surface':
                        temp[i] = T_surface
            return temp
        
        # 生成时间轴（0-50s，间隔0.1s，确保曲线平滑）
        time = np.linspace(0, 50, 501)
        
        # 创建图表（共享X轴，双Y轴）
        fig, ax1 = plt.subplots(figsize=(14, 5))
        
        # 创建共享X轴的第二个Y轴（温度）
        ax2 = ax1.twinx()
        
        # 定义各模态的颜色和标签（保持视觉区分度）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        modes = [
            (air_to_surface, 'Air to Surface'),
            (surface_to_air, 'Surface to Air'),
            (air_to_underwater, 'Air to Underwater'),
            (underwater_to_air, 'Underwater to Air'),
            (surface_to_underwater, 'Surface to Underwater'),
            (underwater_to_surface, 'Underwater to Surface')
        ]
        
        # 绘制各模态的功率和温度曲线
        for i, (mode_func, mode_name) in enumerate(modes):
            # 绘制功率曲线（实线）
            power = mode_func(time)
            ax1.plot(time, power, color=colors[i], label=f'{mode_name} (Power)', linewidth=2.5, alpha=0.8)
            
            # 绘制温度曲线（虚线）
            temp = get_temperature_curve(mode_name, time)
            ax2.plot(time, temp, color=colors[i], linestyle='--', label=f'{mode_name} (Temp)', linewidth=2, alpha=0.8)
        
        # 添加阶段划分背景色（直观区分三个阶段）
        ax1.axvspan(0, 10, alpha=0.1, color='gray', label='Pre-transition')
        ax1.axvspan(10, 40, alpha=0.2, color='yellow', label='Transition')
        ax1.axvspan(40, 50, alpha=0.1, color='green', label='Post-transition')
        
        # 添加基础功率参考线（便于对比初始/目标功率）
        ax1.axhline(y=P_air, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Air Power ({P_air}W)')
        ax1.axhline(y=P_water_surface, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Surface Power ({P_water_surface}W)')
        ax1.axhline(y=P_underwater, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Underwater Power ({P_underwater}W)')
        # 添加2倍P_air参考线（验证水面/水下到空中的功率峰值）
        ax1.axhline(y=2*P_air, color='red', linestyle='-.', linewidth=1.5, alpha=0.6, label=f'2×Air Power ({2*P_air}W)')
        
        # 添加温度参考线
        ax2.axhline(y=T_air, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Air Temp ({T_air}℃)')
        ax2.axhline(y=T_surface, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Surface Temp ({T_surface}℃)')
        ax2.axhline(y=T_underwater, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Underwater Temp ({T_underwater}℃)')
        
        # 设置功率轴（ax1）属性
        ax1.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Power (W)', fontsize=14, fontweight='bold', color='black')
        ax1.set_title('Power and Temperature Switching Process of Different Modes', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 6000)  # 适配2倍P_air（5000W）的显示范围
        ax1.tick_params(axis='y', labelsize=12, colors='black')
        ax1.tick_params(axis='x', labelsize=12)
        
        # 设置温度轴（ax2）属性
        ax2.set_ylabel('Temperature (℃)', fontsize=14, fontweight='bold', color='darkred')
        ax2.set_ylim(-5, 25)  # 温度范围：-5~25℃（覆盖所有场景）
        ax2.tick_params(axis='y', labelsize=12, colors='darkred')
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9, framealpha=0.9, shadow=True)
        
        # 添加阶段标注（增强图表可读性）
        ax1.text(5, ax1.get_ylim()[1] * 0.9, 'Pre-transition\n(0~10s)', ha='center', fontsize=11, 
                backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        ax1.text(25, ax1.get_ylim()[1] * 0.9, 'Transition\n(10~40s)', ha='center', fontsize=11,
                backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        ax1.text(45, ax1.get_ylim()[1] * 0.9, 'Post-transition\n(40~50s)', ha='center', fontsize=11,
                backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        
        # 美化图表边框
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # 调整布局（预留图例空间）
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 创建保存目录
        os.makedirs('./Figures', exist_ok=True)
        
        # 保存为SVG格式（高清无失真，适配论文排版）
        plt.savefig('./Figures/Fig5-6 Power Switching Modes Design.svg', format='svg', dpi=300, bbox_inches='tight')
        
        # 关闭图表释放资源
        plt.close()
        
        print("图表已保存至 ./Figures/Fig5-6 Power Switching Modes Design.svg")

    @staticmethod
    def plot_scenario_profiles():
        """
        绘制Fig5-7：典型场景的功率和温度剖面
        对应原Fig5-7_Profiles.py的功能
        """
        # 定义基础参数
        TOTAL_DURATION = 1800  # 总时长1800s
        SWITCH_DURATION = 50   # 切换模态时长50s
        # 温度常量（℃）
        T_AIR = 0
        T_SURFACE = 20
        T_UNDERWATER = 5
        # 基础模态功率基准值（W）
        P_AIR_BASE = 2500
        P_SURFACE_BASE = 1000
        P_UNDERWATER_BASE = 3000
        # 功率波动百分比参数
        POWER_FLUCTUATION_RATIO_CRUISE = 0.02      # 长航时巡航场景功率波动百分比
        POWER_FLUCTUATION_RATIO_RECON_AIR = 0.02    # 跨域侦察场景空中功率波动百分比
        POWER_FLUCTUATION_RATIO_RECON_SURFACE = 0.02    # 跨域侦察场景水面功率波动百分比
        POWER_FLUCTUATION_RATIO_RECON_UNDERWATER = 0.10    # 跨域侦察场景水下功率波动百分比
        POWER_FLUCTUATION_RATIO_RESCUE = 0.05      # 应急救援场景功率波动百分比
        POWER_FLUCTUATION_MAX = 500      # 所有场景功率波动最大上限
        
        # ===================== 工具函数 =====================
        def generate_bayesian_power(time_points, base_power, scenario_type, mode_type):
            """基于贝叶斯模型生成带随机性的功率曲线"""
            # 不同场景的波动参数
            if scenario_type == 'cruise':  # 长航时巡航：小波动
                std = min(base_power * POWER_FLUCTUATION_RATIO_CRUISE, POWER_FLUCTUATION_MAX)
            elif scenario_type == 'recon':  # 跨域侦察：水下大波动，其他小波动
                if mode_type == 'underwater':
                    std = min(base_power * POWER_FLUCTUATION_RATIO_RECON_UNDERWATER, POWER_FLUCTUATION_MAX)
                elif mode_type == 'air':
                    std = min(base_power * POWER_FLUCTUATION_RATIO_RECON_AIR, POWER_FLUCTUATION_MAX)
                else:  # surface
                    std = min(base_power * POWER_FLUCTUATION_RATIO_RECON_SURFACE, POWER_FLUCTUATION_MAX)
            elif scenario_type == 'rescue':  # 应急救援：整体大波动+峰值
                std = min(base_power * POWER_FLUCTUATION_RATIO_RESCUE, POWER_FLUCTUATION_MAX)
                peak_factor = 1.3  # 峰值系数
            else:
                std = min(base_power * 0.05, POWER_FLUCTUATION_MAX)
        
            # 生成随机波动
            np.random.seed(42)  # 固定随机种子保证可复现
            power_fluctuation = norm.rvs(loc=0, scale=std, size=len(time_points))
            power = base_power + power_fluctuation
        
            # 应急救援场景添加峰值
            if scenario_type == 'rescue':
                peak_indices = np.random.choice(len(time_points), size=int(len(time_points)*0.1), replace=False)
                power[peak_indices] = power[peak_indices] * peak_factor
        
            # 确保功率非负
            power = np.maximum(power, 0)
            return power
        
        def generate_switch_power(time_points, start_power, end_power, switch_type):
            """生成切换模态的功率曲线（参考Fig5-6的切换逻辑）"""
            power = np.zeros_like(time_points, dtype=np.float64)
            for i, ti in enumerate(time_points):
                if ti <= 10:
                    # 过渡前阶段：维持初始功率P1
                    power[i] = start_power
                elif ti <= 20:
                    # 过渡中阶段1：10-20s
                    ratio = (ti - 10) / 10
                    if switch_type == 'air_to_surface':
                        power[i] = start_power + ratio * (1.1 * start_power - start_power)
                    elif switch_type == 'surface_to_air':
                        power[i] = start_power + ratio * (0.9 * start_power - start_power)
                    elif switch_type == 'air_to_underwater':
                        power[i] = start_power
                    elif switch_type == 'underwater_to_air':
                        power[i] = start_power + ratio * (1.1 * start_power - start_power)
                    elif switch_type == 'surface_to_underwater':
                        power[i] = start_power + ratio * (1.05 * start_power - start_power)
                    elif switch_type == 'underwater_to_surface':
                        power[i] = start_power + ratio * (0.95 * start_power - start_power)
                elif ti <= 35:
                    # 过渡中阶段2：20-35s
                    ratio = (ti - 20) / 15
                    if switch_type == 'air_to_surface':
                        power[i] = 1.1 * start_power + ratio * (0.9 * end_power - 1.1 * start_power)
                    elif switch_type == 'surface_to_air':
                        power[i] = 0.9 * start_power + ratio * (2.0 * end_power - 0.9 * start_power)
                    elif switch_type == 'air_to_underwater':
                        power[i] = start_power
                    elif switch_type == 'underwater_to_air':
                        power[i] = 1.1 * start_power + ratio * (2.0 * end_power - 1.1 * start_power)
                    elif switch_type == 'surface_to_underwater':
                        power[i] = 1.05 * start_power + ratio * (1.1 * end_power - 1.05 * start_power)
                    elif switch_type == 'underwater_to_surface':
                        power[i] = 0.95 * start_power + ratio * (0.9 * end_power - 0.95 * start_power)
                elif ti <= 40:
                    # 过渡中阶段3：35-40s
                    ratio = (ti - 35) / 5
                    if switch_type == 'air_to_surface':
                        power[i] = 0.9 * end_power + ratio * (end_power - 0.9 * end_power)
                    elif switch_type == 'surface_to_air':
                        power[i] = 2.0 * end_power + ratio * (end_power - 2.0 * end_power)
                    elif switch_type == 'air_to_underwater':
                        power[i] = end_power
                    elif switch_type == 'underwater_to_air':
                        power[i] = 2.0 * end_power + ratio * (end_power - 2.0 * end_power)
                    elif switch_type == 'surface_to_underwater':
                        power[i] = 1.1 * end_power + ratio * (end_power - 1.1 * end_power)
                    elif switch_type == 'underwater_to_surface':
                        power[i] = 0.9 * end_power + ratio * (end_power - 0.9 * end_power)
                else:
                    # 过渡后阶段：40-50s，维持目标功率P2
                    power[i] = end_power
            power = np.maximum(power, 0)
            return power
        
        def generate_temperature_curve(time_points, mode_sequence, scenario_type):
            """生成温度变化曲线"""
            temp = np.zeros_like(time_points, dtype=np.float64)
            time_idx = 0
            
            # 温度变化速率系数（应急救援更快）
            rate_factor = 1.5 if scenario_type == 'rescue' else 1.0
        
            for segment in mode_sequence:
                seg_type, seg_start, seg_end = segment['type'], segment['start'], segment['end']
                seg_len = seg_end - seg_start
                seg_time = time_points[time_idx:time_idx+seg_len]
                
                if seg_type == 'air':
                    # 空中温度：0℃，带小幅波动
                    seg_temp = np.full(seg_len, T_AIR, dtype=np.float64)
                elif seg_type == 'surface':
                    # 水面温度：20℃，带小幅波动
                    seg_temp = np.full(seg_len, T_SURFACE, dtype=np.float64)
                elif seg_type == 'underwater':
                    # 水下温度：5℃，带小幅波动
                    seg_temp = np.full(seg_len, T_UNDERWATER, dtype=np.float64)
                elif 'switch' in seg_type:
                    # 切换模态温度变化
                    seg_temp = np.zeros(seg_len, dtype=np.float64)
                    switch_time = np.linspace(0, SWITCH_DURATION, seg_len)
                    
                    for i, ti in enumerate(switch_time):
                        if ti <= 10:
                            # 前10s维持初始温度
                            if seg_type == 'air_to_surface_switch':
                                seg_temp[i] = T_AIR
                            elif seg_type == 'surface_to_air_switch':
                                seg_temp[i] = T_SURFACE
                            elif seg_type == 'air_to_underwater_switch':
                                seg_temp[i] = T_AIR
                            elif seg_type == 'underwater_to_surface_switch':
                                seg_temp[i] = T_UNDERWATER
                            elif seg_type == 'surface_to_underwater_switch':
                                seg_temp[i] = T_SURFACE
                            elif seg_type == 'underwater_to_air_switch':
                                seg_temp[i] = T_UNDERWATER
                        elif 10 < ti <= 40:
                            # 过渡阶段线性变化（应急救援速率更快）
                            ratio = ((ti - 10) / 30) * rate_factor
                            ratio = np.clip(ratio, 0, 1)  # 防止超出范围
                            
                            if seg_type == 'air_to_surface_switch':
                                seg_temp[i] = T_AIR + ratio * (T_SURFACE - T_AIR)
                            elif seg_type == 'surface_to_air_switch':
                                seg_temp[i] = T_SURFACE - ratio * (T_SURFACE - T_AIR)
                            elif seg_type == 'air_to_underwater_switch':
                                # 先升后降（25s达20℃）
                                if ti <= 25:
                                    ratio_sub = ((ti - 10) / 15) * rate_factor
                                    seg_temp[i] = T_AIR + ratio_sub * (T_SURFACE - T_AIR)
                                else:
                                    ratio_sub = ((ti - 25) / 15) * rate_factor
                                    seg_temp[i] = T_SURFACE - ratio_sub * (T_SURFACE - T_UNDERWATER)
                            elif seg_type == 'underwater_to_surface_switch':
                                seg_temp[i] = T_UNDERWATER + ratio * (T_SURFACE - T_UNDERWATER)
                            elif seg_type == 'surface_to_underwater_switch':
                                seg_temp[i] = T_SURFACE - ratio * (T_SURFACE - T_UNDERWATER)
                            elif seg_type == 'underwater_to_air_switch':
                                # 先升后降（25s达20℃）
                                if ti <= 25:
                                    ratio_sub = ((ti - 10) / 15) * rate_factor
                                    seg_temp[i] = T_UNDERWATER + ratio_sub * (T_SURFACE - T_UNDERWATER)
                                else:
                                    ratio_sub = ((ti - 25) / 15) * rate_factor
                                    seg_temp[i] = T_SURFACE - ratio_sub * (T_SURFACE - T_AIR)
                        else:
                            # 后10s维持目标温度
                            if seg_type == 'air_to_surface_switch':
                                seg_temp[i] = T_SURFACE
                            elif seg_type == 'surface_to_air_switch':
                                seg_temp[i] = T_AIR
                            elif seg_type == 'air_to_underwater_switch':
                                seg_temp[i] = T_UNDERWATER
                            elif seg_type == 'underwater_to_surface_switch':
                                seg_temp[i] = T_SURFACE
                            elif seg_type == 'surface_to_underwater_switch':
                                seg_temp[i] = T_UNDERWATER
                            elif seg_type == 'underwater_to_air_switch':
                                seg_temp[i] = T_AIR
                else:
                    seg_temp = np.zeros(seg_len, dtype=np.float64)
                
                # 添加温度波动
                np.random.seed(42)
                temp_fluctuation = norm.rvs(loc=0, scale=0.5*rate_factor, size=seg_len)
                seg_temp += temp_fluctuation
                seg_temp = np.clip(seg_temp, -5, 25)  # 温度范围限制
                
                temp[time_idx:time_idx+seg_len] = seg_temp
                time_idx += seg_len
            
            return temp
        
        def build_scenario_profile(scenario_type):
            """构建不同场景的功率和温度剖面"""
            # 生成时间轴（1800s，1s间隔）
            time_points = np.arange(0, TOTAL_DURATION, 1)
            power_profile = np.zeros_like(time_points, dtype=float)
            mode_annotations = []  # 存储模态标注信息
            
            if scenario_type == 'cruise':
                # 长航时巡航：空中(0-600)→切换(600-650)→水面(650-1150)→切换(1150-1200)→空中(1200-1800)
                # 阶段1：空中飞行 0-600s
                air1_time = time_points[0:600]
                air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'cruise', 'air')
                power_profile[0:600] = air1_power
                mode_annotations.append({'type': 'air', 'start': 0, 'end': 600, 'label': 'Air Flight'})
                
                # 阶段2：空中→水面切换 600-650s
                switch1_time = time_points[600:650]
                switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_SURFACE_BASE, 'air_to_surface')
                power_profile[600:650] = switch1_power
                mode_annotations.append({'type': 'air_to_surface_switch', 'start': 600, 'end': 650, 'label': 'Air→Surface Switch'})
                
                # 阶段3：水面航行 650-1150s (500s)
                surface_time = time_points[650:1150]
                surface_power = generate_bayesian_power(surface_time, P_SURFACE_BASE, 'cruise', 'surface')
                power_profile[650:1150] = surface_power
                mode_annotations.append({'type': 'surface', 'start': 650, 'end': 1150, 'label': 'Surface Navigation'})
                
                # 阶段4：水面→空中切换 1150-1200s
                switch2_time = time_points[1150:1200]
                switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
                power_profile[1150:1200] = switch2_power
                mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1150, 'end': 1200, 'label': 'Surface→Air Switch'})
                
                # 阶段5：空中飞行 1200-1800s
                air2_time = time_points[1200:1800]
                air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'cruise', 'air')
                power_profile[1200:1800] = air2_power
                mode_annotations.append({'type': 'air', 'start': 1200, 'end': 1800, 'label': 'Air Flight'})
                
            elif scenario_type == 'recon':
                # 跨域侦察：空中(0-200)→切换(200-250)→水下(250-1300)→切换(1300-1350)→水面(1350-1550)→切换(1550-1600)→空中(1600-1800)
                # 阶段1：空中飞行 0-200s
                air1_time = time_points[0:200]
                air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'recon', 'air')
                power_profile[0:200] = air1_power
                mode_annotations.append({'type': 'air', 'start': 0, 'end': 200, 'label': 'Air Flight'})
                
                # 阶段2：空中→水下切换 200-250s
                switch1_time = time_points[200:250]
                switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_UNDERWATER_BASE, 'air_to_underwater')
                power_profile[200:250] = switch1_power
                mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 200, 'end': 250, 'label': 'Air→Underwater Switch'})
                
                # 阶段3：水下潜航 250-1300s (1050s)
                underwater_time = time_points[250:1300]
                underwater_power = generate_bayesian_power(underwater_time, P_UNDERWATER_BASE, 'recon', 'underwater')
                power_profile[250:1300] = underwater_power
                mode_annotations.append({'type': 'underwater', 'start': 250, 'end': 1300, 'label': 'Underwater Navigation'})
                
                # 阶段4：水下→水面切换 1300-1350s
                switch2_time = time_points[1300:1350]
                switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_UNDERWATER_BASE, P_SURFACE_BASE, 'underwater_to_surface')
                power_profile[1300:1350] = switch2_power
                mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1300, 'end': 1350, 'label': 'Underwater→Surface Switch'})
                
                # 阶段5：水面航行 1350-1550s (200s)
                surface_time = time_points[1350:1550]
                surface_power = generate_bayesian_power(surface_time, P_SURFACE_BASE, 'recon', 'surface')
                power_profile[1350:1550] = surface_power
                mode_annotations.append({'type': 'surface', 'start': 1350, 'end': 1550, 'label': 'Surface Navigation'})
                
                # 阶段6：水面→空中切换 1550-1600s
                switch3_time = time_points[1550:1600]
                switch3_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
                power_profile[1550:1600] = switch3_power
                mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1550, 'end': 1600, 'label': 'Surface→Air Switch'})
                
                # 阶段7：空中飞行 1600-1800s
                air2_time = time_points[1600:1800]
                air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'recon', 'air')
                power_profile[1600:1800] = air2_power
                mode_annotations.append({'type': 'air', 'start': 1600, 'end': 1800, 'label': 'Air Flight'})
                
            elif scenario_type == 'rescue':
                # 应急救援：水面(0-320)→切换(320-370)→空中(370-690)→切换(690-740)→水下(740-1060)→切换(1060-1110)→水面(1110-1430)→切换(1430-1480)→空中(1480-1800)
                # 阶段1：水面航行 0-320s
                surface1_time = time_points[0:320]
                surface1_power = generate_bayesian_power(surface1_time, P_SURFACE_BASE, 'rescue', 'surface')
                power_profile[0:320] = surface1_power
                mode_annotations.append({'type': 'surface', 'start': 0, 'end': 320, 'label': 'Surface Navigation'})
                
                # 阶段2：水面→空中切换 320-370s
                switch1_time = time_points[320:370]
                switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
                power_profile[320:370] = switch1_power
                mode_annotations.append({'type': 'surface_to_air_switch', 'start': 320, 'end': 370, 'label': 'Surface→Air Switch'})
                
                # 阶段3：空中飞行 370-690s (320s)
                air1_time = time_points[370:690]
                air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'rescue', 'air')
                power_profile[370:690] = air1_power
                mode_annotations.append({'type': 'air', 'start': 370, 'end': 690, 'label': 'Air Flight'})
                
                # 阶段4：空中→水下切换 690-740s
                switch2_time = time_points[690:740]
                switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_UNDERWATER_BASE, 'air_to_underwater')
                power_profile[690:740] = switch2_power
                mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 690, 'end': 740, 'label': 'Air→Underwater Switch'})
                
                # 阶段5：水下潜航 740-1060s (320s)
                underwater_time = time_points[740:1060]
                underwater_power = generate_bayesian_power(underwater_time, P_UNDERWATER_BASE, 'rescue', 'underwater')
                power_profile[740:1060] = underwater_power
                mode_annotations.append({'type': 'underwater', 'start': 740, 'end': 1060, 'label': 'Underwater Navigation'})
                
                # 阶段6：水下→水面切换 1060-1110s
                switch3_time = time_points[1060:1110]
                switch3_power = generate_switch_power(np.linspace(0, 50, 50), P_UNDERWATER_BASE, P_SURFACE_BASE, 'underwater_to_surface')
                power_profile[1060:1110] = switch3_power
                mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1060, 'end': 1110, 'label': 'Underwater→Surface Switch'})
                
                # 阶段7：水面航行 1110-1430s (320s)
                surface2_time = time_points[1110:1430]
                surface2_power = generate_bayesian_power(surface2_time, P_SURFACE_BASE, 'rescue', 'surface')
                power_profile[1110:1430] = surface2_power
                mode_annotations.append({'type': 'surface', 'start': 1110, 'end': 1430, 'label': 'Surface Navigation'})
                
                # 阶段8：水面→空中切换 1430-1480s
                switch4_time = time_points[1430:1480]
                switch4_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
                power_profile[1430:1480] = switch4_power
                mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1430, 'end': 1480, 'label': 'Surface→Air Switch'})
                
                # 阶段9：空中飞行 1480-1800s
                air2_time = time_points[1480:1800]
                air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'rescue', 'air')
                power_profile[1480:1800] = air2_power
                mode_annotations.append({'type': 'air', 'start': 1480, 'end': 1800, 'label': 'Air Flight'})
            
            # 生成温度曲线
            temp_profile = generate_temperature_curve(time_points, mode_annotations, scenario_type)
            
            return time_points, power_profile, temp_profile, mode_annotations
        
        # 创建Figures目录
        os.makedirs('./Figures', exist_ok=True)
        
        # 创建子图（3行1列，共享X轴）
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Power and Temperature Profiles of Typical Scenarios', fontsize=18, fontweight='bold', y=0.98)
        
        # 定义场景配置
        scenarios = [
            ('cruise', 'Long-Endurance Cruise', '#1f77b4'),
            ('recon', 'Cross-Domain Reconnaissance', '#ff7f0e'),
            ('rescue', 'Emergency Rescue', '#2ca02c')
        ]
        
        # 绘制每个场景
        for idx, (scenario_type, scenario_label, color) in enumerate(scenarios):
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # 共享X轴的温度轴
            
            # 构建场景剖面
            time, power, temp, modes = build_scenario_profile(scenario_type)
            
            # 绘制功率曲线
            ax1.plot(time, power, color=color, linewidth=1.2, label='Power Demand')
            ax1.fill_between(time, 0, power, color=color, alpha=0.1)
            
            # 绘制温度曲线（虚线）
            ax2.plot(time, temp, color='darkred', linestyle='--', linewidth=1.2, label='Temperature')
            
            # 标注模态阶段
            for mode in modes:
                # 绘制模态背景色
                if 'air' in mode['type'] and 'switch' not in mode['type']:
                    ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightblue')
                elif 'surface' in mode['type'] and 'switch' not in mode['type']:
                    ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightyellow')
                elif 'underwater' in mode['type'] and 'switch' not in mode['type']:
                    ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightgreen')
                elif 'switch' in mode['type']:
                    ax1.axvspan(mode['start'], mode['end'], alpha=0.2, color='orange')
                
                # 添加模态标签（仅标注主要模态）
                if 'switch' not in mode['type']:
                    mid_time = (mode['start'] + mode['end']) / 2
                    ax1.text(mid_time, ax1.get_ylim()[1]*0.7, mode['label'], 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 添加基础功率参考线（便于对比不同模态的功率水平）
            ax1.axhline(y=P_AIR_BASE, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Air Base Power ({P_AIR_BASE}W)')
            ax1.axhline(y=P_SURFACE_BASE, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Surface Base Power ({P_SURFACE_BASE}W)')
            ax1.axhline(y=P_UNDERWATER_BASE, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Underwater Base Power ({P_UNDERWATER_BASE}W)')
            
            # 添加温度参考线
            ax2.axhline(y=T_AIR, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Air Temp ({T_AIR}℃)')
            ax2.axhline(y=T_SURFACE, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Surface Temp ({T_SURFACE}℃)')
            ax2.axhline(y=T_UNDERWATER, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Underwater Temp ({T_UNDERWATER}℃)')
            
            # 设置轴属性
            ax1.set_title(scenario_label, fontsize=14, fontweight='bold', pad=10)
            ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(0, max(power)*1.1)
            ax1.tick_params(axis='y', labelsize=10)
            
            ax2.set_ylabel('Temperature (℃)', fontsize=12, fontweight='bold', color='darkred')
            ax2.set_ylim(-5, 25)
            ax2.tick_params(axis='y', labelsize=10, colors='darkred')
            
            # 美化边框
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            
            # 保存图例信息，但不在单个ax上绘制
            if idx == 0:  # 只在第一个子图收集图例信息
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                fig_legend_handles = lines1 + lines2
                fig_legend_labels = labels1 + labels2
        
        # 设置X轴
        axes[-1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        axes[-1].set_xlim(0, TOTAL_DURATION)
        axes[-1].set_xticks(np.arange(0, TOTAL_DURATION+1, 200))
        axes[-1].tick_params(axis='x', labelsize=10)
        
        # 创建figure级别的共享图例（位于所有Axes之上）
        fig.legend(fig_legend_handles, fig_legend_labels, loc='upper center', fontsize=9, framealpha=0.9, 
                  bbox_to_anchor=(0.5, 0.92), ncol=6)  # 顶部居中，6列布局
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # 调整顶部边距以容纳图例
        
        # 保存SVG文件
        plt.savefig('./Figures/Fig5-7 Profiles.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("图表已保存至 ./Figures/Fig5-7 Profiles.svg")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='EnvUltra环境工具')
    parser.add_argument('--test', action='store_true', help='测试环境功能')
    parser.add_argument('--plot-switch-mode', action='store_true', help='绘制模式切换功率和温度曲线（Fig5-6）')
    parser.add_argument('--plot-scenario-profiles', action='store_true', help='绘制典型场景功率和温度剖面（Fig5-7）')
    
    args = parser.parse_args()
    
    if args.plot_switch_mode:
        # 绘制模式切换功率和温度曲线
        print("=== 绘制Fig5-6: 不同模式切换的功率和温度曲线 ===")
        EnvUltra.plot_switch_mode_power()
        print("=== 绘图完成 ===")
    
    elif args.plot_scenario_profiles:
        # 绘制典型场景功率和温度剖面
        print("=== 绘制Fig5-7: 典型场景功率和温度剖面 ===")
        EnvUltra.plot_scenario_profiles()
        print("=== 绘图完成 ===")
    
    elif args.test:
        # 测试环境功能
        print("=== EnvUltra 测试 ===")
        print(f"支持的场景类型: {EnvUltra.SCENARIO_TYPES}")

        for scenario in EnvUltra.SCENARIO_TYPES:
            print(f"\n--- 测试场景: {scenario} ---")
            env = EnvUltra(scenario_type=scenario)
            print(f"场景类型: {env.scenario_type}")
            print(f"总时长: {env.TOTAL_DURATION}s")
            print(f"步长: {env.step_length}")
            print(f"功率剖面范围: [{env.power_profile.min():.2f}, {env.power_profile.max():.2f}] W")
            print(f"温度剖面范围: [{env.temperature.min():.2f}, {env.temperature.max():.2f}] ℃")
            print(f"模态数量: {len(env.mode_annotations)}")

            s = env.reset()
            print(f"初始观测: {s}")

            total_reward = 0.0
            for step in range(min(10, env.step_length - 1)):
                a_fc = np.random.randint(0, env.N_FC_ACTIONS)
                a_bat = np.random.randint(0, env.N_BAT_ACTIONS)
                a_sc = np.random.randint(0, env.N_SC_ACTIONS)
                action_list = [a_fc, a_bat, a_sc]
                s, r, d, info = env.step(action_list)
                total_reward += r
                if step < 3:
                    print(f"Step {step}: Reward={r:.4f}, Load={info['P_load']:.2f}W, P_fc={info['P_fc']:.2f}W")

            print(f"前10步总奖励: {total_reward:.4f}")
    
    else:
        # 默认显示帮助信息
        parser.print_help()
        print("\n示例用法:")
        print("  测试环境功能: python Env_Ultra.py --test")
        print("  绘制Fig5-6: python Env_Ultra.py --plot-switch-mode")
        print("  绘制Fig5-7: python Env_Ultra.py --plot-scenario-profiles")
