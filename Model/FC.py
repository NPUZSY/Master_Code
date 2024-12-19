import math
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


class Fuel_Cell:
    # https://www.zhihu.com/people/fc-tinker/posts
    time_step = 1e-3
    C = np.array([-3.655e-36, 7.582e-31, -6.636e-26, 3.916e-21, -9.234e-17, 1.636e-12, -1.733e-8, 9.741e-5, 0.2944])

    def __init__(self, time_step=1e-3):
        self.time_step = time_step


    def operatingVoltage(self, i, T=298, p_h2=101, p_o2=101, Ri=0.2, iL=1.4, a=0.5, i0=0.00004):
        def max_reversible_voltage(T, p_h2, p_o2):
            F = 96485
            R = 8.314  # J/mol*K
            Er = 1.229 - 44.43 / (2 * F) * (T - 298) + R * T / (2 * F) * math.log(math.sqrt(p_o2) * p_h2)
            return Er

        def activationLoss(Current, i_0, Temperature, a):
            F = 96485
            R = 8.314  # J/mol*K
            vLoss = (R * Temperature / (a * F)) * math.log(Current / i_0)
            return vLoss

        # Current:A/cm2;  Ri inclueds R_ionic, R_elect, R_cont: Ωcm2
        # typically, total resistance is from 0.1~0.2Ωcm2
        def ohmicLoss(Current, Ri):
            vLoss = Current * Ri
            return vLoss

        def concentrationLoss(Current, iL, Temperature):
            F = 96485
            R = 8.314  # J/mol*K
            n = 2
            vLoss = (R * Temperature / (n * F)) * math.log(iL / (iL - Current))
            return vLoss

        Er = max_reversible_voltage(T, p_h2, p_o2)
        V_ohmic = ohmicLoss(i, Ri)
        V_act = activationLoss(i, i0, T, a)
        V_conc = concentrationLoss(i, iL, T)
        E = Er - V_act - V_ohmic - V_conc
        return E

    def hydrogen_consumption(self, i):
        voltage = self.operatingVoltage(i)
        power = i * voltage
        efficiency = 0
        for i in range(9):
            efficiency += self.C[i] * power ** i
        print(f"效率: {efficiency}")
        E_hd = 143000
        consumption = power * self.time_step / (efficiency * E_hd)
        print(f"消耗: {consumption}g 氢气")
        return consumption

    def temp_power_to_efficiency(self, temperature=25, power=1000):
        return


class FCS:
    # C = np.array([-3.655e-36, 7.582e-31, -6.636e-26, 3.916e-21, -9.234e-17, 1.636e-12, -1.733e-8, 9.741e-5, 0.2944])
    C_new = np.array([-5.571e-16,
                      3.467e-13,
                      -9.103e-11,
                      1.315e-08,
                      -1.14e-06,
                      6.059e-05,
                      -0.001925,
                      0.03247,
                      0.2944])

    def __init__(self):
        self.Eng_fuel_func = np.poly1d(self.C_new)  # 初始化效率计算函数

    def cal_efficiency_new(self):
        data_num = 1001
        power = np.linspace(0, 30, num=data_num)  # 初始化横坐标
        # eff = self.Eng_fuel_func(power)          # 计算功率对应的效率

        eff = np.zeros(data_num)
        for i in range(power.size):
            eff[i] = self.Eng_fuel_func(power[i] / 300 * 1000)

        fig, ax = plt.subplots()  # 绘制曲线

        ax.plot(power, eff)
        ax.set_xlabel('Output Power / kW', color='black')
        ax.set_ylabel('Efficiency', color='black')
        plt.show()
        fig.savefig('../Figures/FC_Efficiency.png', dpi=1200)
    # def cal_efficiency(self):
    #     power = np.linspace(0.1, 3.5, num=1999)
    #     efficiency = 0
    #     for i in range(9):
    #         efficiency += self.C[i] * power ** (8 - i)
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(power, efficiency)
    #     plt.show()


def main():
    FC = Fuel_Cell()

    x = np.linspace(0.001, 1.399, num=1399)

    y = np.array([FC.operatingVoltage(x_) for x_ in list(x)])

    power = x * y

    print(x)
    print(y)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x, power)
    ax.set_xlabel('Current Density')
    ax.set_ylabel('Voltage / V')
    plt.show()
    print(np.where(power == power.max()))

    print(FC.hydrogen_consumption(1.3))


if __name__ == '__main__':
    main()
    # F = FCS()
    # F.cal_efficiency()
    # F.cal_efficiency_new()
