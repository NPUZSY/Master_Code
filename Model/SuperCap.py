import math
import numpy as np


class SuperCapacitor:
    """
    超级电容模型（等效电容 + ESR）
    - 供你的环境 Env_FC_Li 调用
    - 接口对齐 BatterySimple / BatteryPlot 的最简调用形式
    """

    def __init__(self,
                 voltage_init=48.0,  # 初始电压（V）
                 capacitance=50.0,  # 电容（F） 论文数值：几十 F 级
                 esr=0.08,  # 等效串联电阻 ESR（Ohm）
                 v_min=24.0,  # 最低工作电压（V）
                 v_max=48.0,  # 最高工作电压（V）
                 time_step=1.0):  # ❗ 优化: 将时间步长（秒）设置为 1.0s，与环境步长对齐

        self.C = capacitance
        self.esr = esr
        self.v_cap = voltage_init
        self.v_min = v_min
        self.v_max = v_max
        self.time_step = time_step  # 现在是 1.0s

        # 额外信息（类似 Battery）
        self.power = 0.0  # 超级电容提供/吸收功率（W）
        self.i_sc = 0.0  # 电流（A）
        self.soc = (self.v_cap - v_min) / (v_max - v_min)  # 定义等效 SOC∈[0,1]

    def _update_soc(self):
        """根据电压重新计算 SOC"""
        self.soc = (self.v_cap - self.v_min) / (self.v_max - self.v_min)
        self.soc = max(0.0, min(1.0, self.soc))

    def output(self, p_sc=0.0):
        """
        超级电容输出功能：
        输入:
            p_sc: 功率（W），正值放电（提供功率），负值充电

        输出:
            i_sc: 电流（A）
            v_cap: 更新后的电压
            soc: 超级电容 SOC-like
            p_sc: 实际输出功率
        """

        self.power = p_sc
        actual_p_sc = p_sc # 默认实际功率等于指令功率

        # 1) 电流计算 (基于 t 时刻的电压近似)
        if self.v_cap <= 1e-6:
            self.i_sc = 0.0
        else:
            # ❗ 注意：I = P / V_cap，在 1.0s 大步长下为粗糙近似
            self.i_sc = p_sc / self.v_cap

        # 2) 电压动态更新 (dv 是基于 1.0s 的变化)
        # dv = -(I/C)*dt
        dv_cap_only = -(self.i_sc / self.C) * self.time_step # time_step = 1.0s

        # 3) 考虑 ESR 压降
        v_drop = self.i_sc * self.esr

        # 4) 更新电压：V_new = V_old + dV_cap - V_drop
        self.v_cap += dv_cap_only - v_drop

        # 5) 限幅保护
        if self.v_cap > self.v_max:
            self.v_cap = self.v_max
            self.i_sc = 0
            actual_p_sc = 0
        if self.v_cap < self.v_min:
            self.v_cap = self.v_min
            self.i_sc = 0
            actual_p_sc = 0

        # 更新 SOC
        self._update_soc()

        return self.i_sc, self.v_cap, self.soc, actual_p_sc


# 用于快速测试
if __name__ == "__main__":
    # 测试时，步长现在是 1.0s，每一步模拟一个 RL 周期
    sc = SuperCapacitor()
    print("SuperCapacitor with time_step = 1.0s")
    for i in range(10):
        # ❗ 注意：这里的 output 应该被环境调用 10 次，而不是 1000 次
        sc.output(500)  # 放电 500W
        print(f"Step {i+1}: i={sc.i_sc:.2f}A  v={sc.v_cap:.2f}V  soc={sc.soc:.3f}")