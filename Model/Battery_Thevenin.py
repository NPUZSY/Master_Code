import matplotlib.pyplot as plt
import thevenin
import numpy as np

stressors = {'q_dis': 1.}


def calc_xa(soc: float) -> float:
    return 8.5e-3 + soc * (7.8e-1 - 8.5e-3)


def calc_Ua(soc: float) -> float:
    xa = calc_xa(soc)
    Ua = 0.6379 + 0.5416 * np.exp(-305.5309 * xa) \
         + 0.0440 * np.tanh(-1. * (xa - 0.1958) / 0.1088) \
         - 0.1978 * np.tanh((xa - 1.0571) / 0.0854) \
         - 0.6875 * np.tanh((xa + 0.0117) / 0.0529) \
         - 0.0175 * np.tanh((xa - 0.5692) / 0.0875)

    return Ua


def normalize_inputs(soc: float, T_cell: float) -> dict:
    inputs = {
        'T_norm': T_cell / (273.15 + 35.),
        'Ua_norm': calc_Ua(soc) / 0.123,
    }
    return inputs


def ocv_func(soc: float) -> float:
    coeffs = np.array([
        1846.82880284425, -9142.89133579961, 19274.3547435787, -22550.631463739,
        15988.8818738468, -7038.74760241881, 1895.2432152617, -296.104300038221,
        24.6343726509044, 2.63809042502323,
    ])
    return np.polyval(coeffs, soc)


def R0_func(soc: float, T_cell: float) -> float:
    inputs = normalize_inputs(soc, T_cell)
    T_norm = inputs['T_norm']
    Ua_norm = inputs['Ua_norm']

    b = np.array([4.07e12, 23.2, -16., -47.5, 2.62])

    R0 = b[0] * np.exp(b[1] / T_norm ** 4 * Ua_norm ** (1 / 4)) \
         * np.exp(b[2] / T_norm ** 4 * Ua_norm ** (1 / 3)) \
         * np.exp(b[3] / T_norm ** 0.5) \
         * np.exp(b[4] / stressors['q_dis'])

    return R0


def R1_func(soc: float, T_cell: float) -> float:
    inputs = normalize_inputs(soc, T_cell)
    T_norm = inputs['T_norm']
    Ua_norm = inputs['Ua_norm']

    b = np.array([2.84e-5, -12.5, 11.6, 1.96, -1.67])

    R1 = b[0] * np.exp(b[1] / T_norm ** 3 * Ua_norm ** (1 / 4)) \
         * np.exp(b[2] / T_norm ** 4 * Ua_norm ** (1 / 4)) \
         * np.exp(b[3] / stressors['q_dis']) \
         * np.exp(b[4] * soc ** 4)

    return R1


def C1_func(soc: float, T_cell: float) -> float:
    inputs = normalize_inputs(soc, T_cell)
    T_norm = inputs['T_norm']
    Ua_norm = inputs['Ua_norm']

    b = np.array([19., -3.11, -27., 36.2, -0.256])

    C1 = b[0] * np.exp(b[1] * soc ** 4) \
         * np.exp(b[2] / T_norm ** 4 * Ua_norm ** (1 / 2)) \
         * np.exp(b[3] / T_norm ** 3 * Ua_norm ** (1 / 3)) \
         * np.exp(b[4] / stressors['q_dis'] ** 3)

    return C1


class BatterySimple:
    def __init__(self, soc=0.5, capacity=75.):
        self.soc = soc  # 电池SoC
        self.soc_ref = 0.6
        self.capacity_total = capacity
        self.capacity = self.capacity_total * self.soc
        self.params = {
            'num_RC_pairs': 1,
            'soc0': self.soc,
            'capacity': self.capacity_total,
            'mass': 1.9,
            'isothermal': False,
            'Cp': 745.,
            'T_inf': 300.,
            'h_therm': 12.,
            'A_therm': 1.,
            'ocv': ocv_func,
            'R0': R0_func,
            'R1': R1_func,
            'C1': C1_func,
        }
        self.model = thevenin.Model(self.params)
        self.expr = thevenin.Experiment()
        self.expr.add_step('power_W', 0., (1., 1.))
        self.soln = self.model.run(self.expr)

    def work(self, output_power=0):
        # 输出的功率作用在容量上
        soc_ = self.soln.vars['soc'][-1]
        if ((soc_ < 0.01) and (output_power > 0)) or ((soc_ > 0.99) and (output_power < 0)):
            output_power = 0
        self.expr.add_step('power_W', output_power, (1, 1))
        self.soln = self.model.run(self.expr)
        self.soc = self.soln.vars['soc'][-1]
        soc_diff = soc_ - self.soc
        soc_err = self.soc - self.soc_ref
        return soc_diff, soc_err, output_power

    def plot(self):
        self.soln.plot('time_s', 'soc')
        self.soln.plot('time_s', 'voltage_V')
        plt.show()


if __name__ == '__main__':
    battery = BatterySimple(0.02, 10)
    print(battery.work(15))
    print(battery.work(-15))
    print(battery.work(-15))
    print(battery.work(15))
    battery.plot()

