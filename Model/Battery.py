import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

class BatteryPlot:
    def __init__(self, soc=0.5, capacity=1540 * 3600, time_step=1e-3):
        # 电池容量为1.54KWh，换算成W*s = J
        self.soc = soc  # 电池SoC
        self.soc_ref = 0.6
        self.i_b = 0  # 电流
        self.p_b = 0  # 功率
        self.capacity = capacity  # 容量
        self.time_step = time_step  # 步长

        self.soc_x = np.linspace(0, 1, num=11)  # 用于坐标表示的SoC，并非电池SoC
        self.Res_charge = np.array([0.7, 0.63, 0.48, 0.4, 0.385, 0.375, 0.35, 0.37, 0.38, 0.37, 0.36])
        self.Res_discharge = np.array([0.7, 0.631, 0.42, 0.4, 0.38, 0.37, 0.35, 0.355, 0.36, 0.38, 0.4])
        self.Voltage = np.array([202, 210, 212, 215, 219, 221, 222, 223, 225, 231, 237])

        # 插值函数
        self.Res_charge_f = interp1d(self.soc_x, self.Res_charge, kind='cubic')
        self.Res_discharge_f = interp1d(self.soc_x, self.Res_discharge, kind='cubic')
        self.Voltage_f = interp1d(self.soc_x, self.Voltage, kind='cubic')

    def plot(self):
        fig, ax = plt.subplots()

        # 设置左侧Y轴（对应电阻曲线）刻度标签颜色与电阻曲线颜色一致
        color_resistance = '#1f77b4'
        ax.plot(self.soc_x * 100, self.Res_charge, "--", color=color_resistance)
        ax.plot(self.soc_x * 100, self.Res_discharge, "-", color=color_resistance)
        ax.set_ylabel('Resistance / Ω', color=color_resistance)
        ax.tick_params(axis='y', labelcolor=color_resistance)
        ax.set_xlabel('SOC / %')
        ax.spines['right'].set_visible(False)
        ax.legend(['Charge Resistance', 'Discharge Resistance'], loc=2)

        z_ax = ax.twinx()  # 创建与轴群ax共享x轴的轴群z_ax

        # 设置右侧Y轴（对应电压曲线）刻度标签颜色与电压曲线颜色一致
        color_voltage = '#ff7f0e'
        z_ax.plot(self.soc_x * 100, self.Voltage, color=color_voltage)
        z_ax.set_ylabel('Voltage / V', color=color_voltage)
        z_ax.tick_params(axis='y', labelcolor=color_voltage)
        z_ax.legend(["Voltage"], loc=1)

        plt.xlim(0, 100)
        ax.set_ylim(0.3, 0.8)
        z_ax.set_ylim(200, 245)
        fig.savefig('../Figures/Battery_Paras.png', dpi=1200)
        plt.show()

    def output(self, u_b=0, p_b=0, charge=True):

        if charge:
            Res = self.Res_charge_f(self.soc)
        else:
            Res = self.Res_discharge_f(self.soc)
        u_oc = self.Voltage_f(self.soc)
        self.p_b = p_b
        # self.p_b = u_b * self.i_b
        self.i_b = (u_oc - math.sqrt(u_oc ** 2 - 4 * Res * self.p_b)) / (2 * Res)
        dSoC = self.i_b * self.time_step / self.capacity
        self.soc = self.soc - dSoC

        # 更新开路电压
        u_oc = self.Voltage_f(self.soc)
        print(f"电池电流：{self.i_b} 开路电压：{u_oc} SoC：{self.soc} 功率：{self.p_b} SoC变化：{dSoC}")
        return self.i_b, u_oc, self.soc, self.p_b


class BatterySimple:
    def __init__(self, soc=0.5, capacity=1540 * 3600):
        self.soc = soc  # 电池SoC
        self.soc_ref = 0.6
        self.capacity_total = capacity
        self.capacity = self.capacity_total * self.soc

    def work(self, power=0):
        # 输出的功率作用在容量上
        self.capacity -= power
        # SoC不能超过上下限
        if self.capacity >= self.capacity_total:
            self.capacity = self.capacity_total
            power = 0
        if self.capacity <= 0:
            self.capacity = 0
            power = 0
        soc_ = self.capacity / self.capacity_total
        soc_diff = soc_ - self.soc
        self.soc = soc_
        soc_err = self.soc - self.soc_ref
        return soc_diff, soc_err, power


def main():
    b = BatteryPlot()
    b.plot()
    # for i in range(10000):
    #     b.output(300, 3000)


if __name__ == '__main__':
    main()
