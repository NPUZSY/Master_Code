import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import norm

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

    SCENARIO_TYPES = ['cruise', 'recon', 'rescue']

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


if __name__ == "__main__":
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
