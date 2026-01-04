import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import time  # 保持 time 模块导入，以便测试脚本可以运行

# 请确认 Model 中存在这些模块文件并导出相应类
from Model.Battery import BatterySimple
from Model.FC import FCS
from Model.SuperCap import SuperCapacitor
from Scripts.Power_Profile import UAV_Load


class Envs(gym.Env):
    """
    三源耦合环境（FC + Battery + SuperCap）
    - 适配三智能体 I-DQN 架构的动作输入（现为动作列表）。
    - 核心修改1：锂电池功率由智能体动作直接决定，超级电容补偿功率差值
    - 核心修改2：超级电容满充/放空时继续充/放电，按剩余功率的10倍惩罚
    """

    def __init__(self):
        super().__init__()

        # -------------------
        # 系统常量 / 可调参数
        # -------------------
        self.dt = 1.0  # 控制周期 1s
        self.calorific_value = 143000.0  # J/g (氢气热值)
        self.Q_H2_eq = 142000.0  # J/g 等效换算（142 MJ/kg -> 142000 J/g）

        # Fuel cell / battery / supercap limits (可按需修改)
        self.P_FC_MAX = 5000.0  # W
        self.P_FC_MIN = 0.0

        # Battery max charge/discharge power (绝对值)
        self.P_BAT_MAX = 5000.0  # W  (根据你电池模型调整)
        # Supercap max instantaneous power (吸放)
        self.P_SC_MAX = 2000.0  # W

        # -------------------
        # 奖励权重
        # -------------------
        self.w1 = -200
        self.w2 = -0.1
        self.w3 = -0.1
        # 新增：超级电容过充/过放惩罚权重（功率的10倍）
        self.w_sc_punish = 10
        self.minmatch_punish = 10
        # 注意：这里的断言是判断 w1+w2+w3 + 1.0 是否接近于 0 (即 w1+w2+w3 约为 -1)
        # 原代码中的断言逻辑存在问题，这里将其简化为检查和是否为负数且非零
        if self.w1 + self.w2 + self.w3 >= 0:
            print("警告：奖励权重之和非负，可能导致训练异常。")

        # -------------------
        # 环境工况（载荷 / 温度）
        # -------------------
        # 修正导入路径
        try:
            loads_data = UAV_Load.get_loads()
            self.temperature = loads_data[0]
            self.loads = loads_data[1]
        except Exception:
            # 如果导入失败，提供一个回退数据防止崩溃
            print("警告: 无法加载 UAV_Load, 使用默认值。")
            self.temperature = np.array([25.0] * 600)
            self.loads = np.array([1000.0] * 600)

        self.step_length = len(self.loads)

        # -------------------
        # 能源模块实例
        # -------------------
        self.battery = BatterySimple()
        self.fuel_cell = FCS()
        self.supercap = SuperCapacitor()

        # -------------------
        # 动作空间定义 (32 x 20 x 2)
        # -------------------
        self.K_FC_MIN = -15
        self.K_FC_MAX = 16
        self.K_BAT_MIN = -20
        self.K_BAT_MAX = 19

        self.N_FC_ACTIONS = self.K_FC_MAX - self.K_FC_MIN + 1  # 32
        self.N_BAT_ACTIONS = self.K_BAT_MAX - self.K_BAT_MIN + 1  # 40
        self.N_SC_ACTIONS = 2  # 2

        # ❗ 注意：N_ACTIONS 仅用于兼容旧的单整数动作空间或日志，现已无实际意义
        self.N_ACTIONS = self.N_FC_ACTIONS * self.N_BAT_ACTIONS * self.N_SC_ACTIONS  # 1280

        # 保持 Dict 结构用于内部验证和兼容 gym.Env，但实际输入为单个整数
        self.action_space = spaces.Dict({
            'fc': spaces.Discrete(self.N_FC_ACTIONS),
            'bat': spaces.Discrete(self.N_BAT_ACTIONS),
            'sc': spaces.Discrete(self.N_SC_ACTIONS)
        })

        # -------------------
        # 观察空间: [P_load, temperature, P_fc, P_bat, P_sc, soc_bat, soc_sc]
        # -------------------
        self.observation_space = spaces.Box(
            low=np.array([0., -100., 0., -self.P_BAT_MAX, -self.P_SC_MAX, 0., 0.], dtype=np.float32),
            high=np.array([80000., 200., self.P_FC_MAX, self.P_BAT_MAX, self.P_SC_MAX, 1., 1.], dtype=np.float32),
            dtype=np.float32
        )

        # -------------------
        # 内部状态
        # -------------------
        self.time_stamp = 0
        self.power_fc = 0.0  # 当前 FC 输出（W）
        self.r_fc_accum = 0.0  # FC 超限惩罚累计
        self.punish_step = 1.0  # 每步累积值（可调）
        self.punish_decay = 0.5  # 衰减量（当恢复安全时）
        # 新增：超级电容过充/过放惩罚累计
        self.r_sc_punish = 0.0
        self.reset()

    # -------------------
    # helper: action index -> physical value
    # -------------------
    def _fc_delta_from_index(self, idx):
        # FC 动作索引 0..31 对应 k in [-15,..,16]
        k = self.K_FC_MIN + int(idx)
        # 变化率步长为 0.001 * P_FC_MAX
        delta = k * 0.001 * self.P_FC_MAX
        return float(delta)

    def _bat_power_from_index(self, idx):
        # Bat 动作索引 0..19 对应 k in [-10..9]
        k = self.K_BAT_MIN + int(idx)
        # 功率步长为 0.1 * P_BAT_MAX
        p = k * 0.05 * self.P_BAT_MAX
        return float(p)

    # -------------------
    # 重置 (保持不变，新增超级电容惩罚重置)
    # -------------------
    def reset(self, **kwargs):
        self.time_stamp = 0
        # 重置模块（调用各自构造器）
        self.battery = BatterySimple()
        self.fuel_cell = FCS()
        self.supercap = SuperCapacitor()
        self.power_fc = 0.0
        self.r_fc_accum = 0.0
        # 新增：重置超级电容过充/过放惩罚
        self.r_sc_punish = 0.0

        P_load = float(self.loads[0])
        T_env = float(self.temperature[0]) if len(self.temperature) > 0 else 0.0

        # get initial battery & supercap soc
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

    # -------------------
    # STEP (核心修改：锂电池功率由动作直接决定，超级电容补偿差值+过充/过放惩罚)
    # -------------------
    def step(self, action_list):
        """
        action_list: 包含三个动作索引的列表/数组：[a_fc, a_bat, a_sc]
        核心修改1：锂电池功率(P_bat_final)直接由智能体动作决定（仅做功率上下限约束）
        核心修改2：超级电容补偿「负载需求 - 燃料电池功率 - 锂电池功率」的功率差值
        核心修改3：超级电容满充(SOC=1)继续充电/放空(SOC=0)继续放电，按剩余功率10倍惩罚
        """

        # 1) 直接从列表读取三个动作索引
        a_fc = int(action_list[0])
        a_bat = int(action_list[1])
        a_sc = int(action_list[2])

        # 2) 封装成 Dict 供后续逻辑使用
        action_decoded = {
            'fc': a_fc,
            'bat': a_bat,
            'sc': a_sc
        }

        # 当前负载/温度 (使用上一个时刻的 observation)
        P_load = float(self.current_observation[0])
        T_env = float(self.current_observation[1])

        # 1) 将动作映射到物理量
        delta_P_fc = self._fc_delta_from_index(action_decoded['fc'])
        P_bat_cmd = self._bat_power_from_index(action_decoded['bat'])  # 智能体选择的锂电池功率
        sc_on = bool(int(action_decoded['sc']) == 1)
        # print(sc_on)
        

        # 2) FC 输出随动作变化（∆P_fc），但受速率与上下限约束
        self.power_fc = float(np.clip(self.power_fc + delta_P_fc, self.P_FC_MIN, self.P_FC_MAX))

        # 3) 锂电池功率：直接使用智能体动作值（仅做上下限约束）
        P_bat_final = float(np.clip(P_bat_cmd, -self.P_BAT_MAX, self.P_BAT_MAX))

        # 4) 超级电容补偿功率差值：计算负载需求与 FC+电池 输出的差值
        power_diff = P_load - self.power_fc - P_bat_final  # 需补偿的功率差值

        # sc_on = ~(power_diff == 0)
        
        # 超级电容根据开关状态和功率限制补偿差值
        if sc_on:
            P_sc = float(np.clip(power_diff, -self.P_SC_MAX, self.P_SC_MAX))  # 补偿差值（受功率限制）
        else:
            P_sc = 0.0  # 超级电容关闭时不补偿

        # 7) 将最终功率下达到各模块，更新模块状态
        # Battery: 使用其 work 接口（传入智能体选定的功率）
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
            # 保险回退：若接口不匹配，则直接近似更新 soc
            try:
                soc_prev = float(self.battery.soc)
                energy_delta = P_bat_final * self.dt  # J
                cap_total = getattr(self.battery, "capacity_total", getattr(self.battery, "capacity", 1.0))
                soc_new = max(0.0, min(1.0, soc_prev - energy_delta / (cap_total + 1e-9)))
                soc_diff = soc_prev - soc_new
                soc_err = soc_new - getattr(self.battery, "soc_ref", 0.6)
                self.battery.soc = soc_new
                actual_bat_power = P_bat_final
            except Exception:
                soc_diff, soc_err, actual_bat_power = 0.0, 0.0, P_bat_final

        # Supercap: 调用 output 接口（传入补偿的功率值）
        try:
            i_sc, v_sc, soc_sc, actual_p_sc = self.supercap.output(P_sc)
        except Exception:
            actual_p_sc = P_sc
            try:
                # 确保 SuperCapacitor 模块有 soc 属性
                if hasattr(self.supercap, 'soc'):
                    soc_sc = self.supercap.soc
                else:
                    soc_sc = 0.5
            except Exception:
                soc_sc = 0.5

        # ----------------------------
        # 新增：超级电容过充/过放惩罚计算
        # ----------------------------
        # 重置当前步惩罚
        current_sc_punish = 0.0
        # 获取超级电容SOC（限制在0~1范围）
        soc_sc_clamped = np.clip(soc_sc, 0.0, 1.0)
        # P_sc > 0: 超级电容放电；P_sc < 0: 超级电容充电
        if sc_on:
            # 情况1：SOC=1 且 继续充电（P_sc < 0）
            if np.isclose(soc_sc_clamped, 1.0) and P_sc < 0:
                current_sc_punish = abs(P_sc) * self.w_sc_punish
            # 情况2：SOC=0 且 继续放电（P_sc > 0）
            elif np.isclose(soc_sc_clamped, 0.0) and P_sc > 0:
                current_sc_punish = abs(P_sc) * self.w_sc_punish
        # 累计惩罚
        self.r_sc_punish += current_sc_punish

        # Fuel cell: FC 消耗和效率估计
        P_fc = float(self.power_fc)
        eta_fc = None
        try:
            if hasattr(self.fuel_cell, "Eng_fuel_func"):
                try:
                    # 尝试调用不同的功率单位
                    eta_fc = float(self.fuel_cell.Eng_fuel_func(P_fc / 1000.0))
                except Exception:
                    eta_fc = float(self.fuel_cell.Eng_fuel_func(P_fc))
            elif hasattr(self.fuel_cell, "cal_efficiency"):
                eta_fc = float(self.fuel_cell.cal_efficiency(P_fc))
        except Exception:
            eta_fc = None

        if eta_fc is None or math.isnan(eta_fc) or eta_fc <= 0:
            eta_fc = 0.45  # 默认燃料电池效率

        # 变换器效率
        eta_conv = 0.95

        # ----------------------------
        # 等效氢耗计算（g）
        # ----------------------------
        C_fc = 0.0
        C_bat = 0.0
        if P_fc > 0:
            C_fc = (P_fc * self.dt) / (max(1e-6, eta_fc * eta_conv) * self.calorific_value)  # g
        # battery: use actual_bat_power (正为放电)
        C_bat = (actual_bat_power * self.dt) / (eta_conv * self.Q_H2_eq)  # g

        # ----------------------------
        # 安全惩罚项
        # ----------------------------
        if P_fc > 0.9 * self.P_FC_MAX:
            self.r_fc_accum += self.punish_step
        else:
            self.r_fc_accum = max(0.0, self.r_fc_accum - self.punish_decay)

        r_fc = float(self.r_fc_accum)

        # battery soc
        try:
            soc_b = float(self.battery.soc)
        except Exception:
            soc_b = 0.5

        if soc_b < 0.2 or soc_b > 0.8:
            r_bat = 1.0  # 固定惩罚值
        else:
            r_bat = 0.0
        
        # 偏离0.6的惩罚
        r_bat += abs(soc_b - 0.6) * 5

        # ----------------------------
        # 匹配误差（保持原有逻辑）
        # ----------------------------
        # 完全没匹配上的功率和又超级电容补充的功率
        power_loss = abs(P_load - self.power_fc - actual_bat_power - actual_p_sc)  
        r_match = current_sc_punish + power_loss * self.minmatch_punish

        # ----------------------------
        # 总奖励（新增超级电容过充/过放惩罚项）
        # ----------------------------
        reward = float(
            self.w1 * (C_fc + C_bat) + 
            self.w2 * (r_fc + r_bat) + 
            self.w3 * r_match
        ) / self.step_length *10

        # ----------------------------
        # 时间推进与终止（保持原有逻辑）
        # ----------------------------
        self.time_stamp += 1
        done = bool(self.time_stamp >= len(self.loads) - 1)

        # 下一个时刻载荷 / 温度
        if not done:
            next_load = float(self.loads[self.time_stamp])
            next_temp = float(self.temperature[self.time_stamp]) if len(self.temperature) > self.time_stamp else 0.0
        else:
            next_load = 0.0
            next_temp = 0.0

        # 更新 observation
        self.current_observation = np.array([
            next_load,
            next_temp,
            self.power_fc,
            actual_bat_power,
            actual_p_sc,
            soc_b,
            soc_sc
        ], dtype=np.float32)

        # info for logging（新增超级电容惩罚相关字段）
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
            # 新增字段
            "soc_sc": soc_sc_clamped,
            "current_sc_punish": current_sc_punish,
            "total_sc_punish": self.r_sc_punish
        }

        return self.current_observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":

    print("--- 🚀 Environment Step Speed Test ---")
    env = Envs()

    # 设定测试步数
    NUM_TEST_STEPS = 100

    # 初始化环境
    s = env.reset()
    total_step_time = 0.0

    # 确保测试步数不超过环境的最大步长
    max_steps_to_test = min(NUM_TEST_STEPS, env.step_length - 1)

    print(f"Testing {max_steps_to_test} steps (Max Episode Length: {env.step_length})")

    # 开始计时
    start_time_total = time.time()

    # 定义动作空间大小
    N_FC = env.N_FC_ACTIONS
    N_BAT = env.N_BAT_ACTIONS
    N_SC = env.N_SC_ACTIONS

    for t in range(max_steps_to_test):
        # 模拟训练代码传入动作列表
        a_fc = np.random.randint(0, N_FC)
        a_bat = np.random.randint(0, N_BAT)
        a_sc = np.random.randint(0, N_SC)

        action_list = [a_fc, a_bat, a_sc]

        # 测量单步时间
        step_start_time = time.time()
        # ❗ 注意：这里传入的是 action_list
        s, r, d, info = env.step(action_list)
        step_end_time = time.time()

        total_step_time += (step_end_time - step_start_time)

        # 仅打印前几步的详细信息（新增超级电容惩罚信息）
        if t < 5:
            print(f"Step {t}: Action={action_list}, Reward={r:.4f}, P_fc={info.get('P_fc'):.2f} W, P_bat={info.get('P_bat'):.2f} W, P_sc={info.get('P_sc'):.2f} W, SOC_B={s[-2]:.4f}, SOC_SC={info.get('soc_sc'):.4f}, SC_Punish={info.get('current_sc_punish'):.2f}")

        if d:
            break

    end_time_total = time.time()

    # 统计结果
    num_executed_steps = t + 1
    total_duration = end_time_total - start_time_total
    avg_step_time = total_step_time / num_executed_steps

    # 计算估算的单回合时间 (基于环境的完整步长)
    estimated_episode_time_s = avg_step_time * env.step_length

    print("\n" + "=" * 40)
    print("        📊 Test Results 📊")
    print("=" * 40)
    print(f"1. Total Steps Tested: {num_executed_steps}")
    print(f"2. Total Test Duration: {total_duration:.2f} seconds")
    print(f"3. ⚡️ Average Time per Step: {avg_step_time * 1000:.2f} ms")
    print(f"4. ⏳ Estimated Episode Time (Full {env.step_length} Steps): {estimated_episode_time_s:.2f} seconds ({estimated_episode_time_s / 60:.2f} minutes)")
    print(f"5. 🔋 Total SuperCap Punish: {env.r_sc_punish:.2f}")
    print("=" * 40)