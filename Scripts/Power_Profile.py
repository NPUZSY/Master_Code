import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman', 'SimHei'
plt.rcParams['font.size'] = 12

# 定义高斯噪声参数（可根据需要调整标准差控制噪声强度）
TEMPERATURE_NOISE_STD = 3.0  # 温度噪声标准差
POWER_NOISE_STD = 80.0  # 功率噪声标准差

# 定义起飞阶段数据
takeoff_time = np.linspace(0, 200, 200)
takeoff_temperature1 = np.linspace(-75, -75, 150)
takeoff_temperature2 = np.linspace(-75, 25, 50)
takeoff_temperature = np.concatenate((takeoff_temperature1, takeoff_temperature2))

# 模拟起飞阶段功率变化
takeoff_time1 = np.linspace(0, 150, 150)
takeoff_time2 = np.linspace(150, 170, 20)
takeoff_time3 = np.linspace(170, 200, 30)
takeoff_power = np.concatenate(
    (takeoff_time1 * 0 + 4000, takeoff_time2 * 40 - 2000, (takeoff_time3 - 170) * (-120) + 4600))

# 定义巡航阶段数据
cruise_time = np.linspace(200, 400, 200)
cruise_temperature = np.full(200, 25)
cruise_power = np.full(200, 1000)

# 定义降落阶段数据
landing_time = np.linspace(400, 600, 200)
landing_temperature1 = np.linspace(25, 5, 30)
landing_temperature2 = np.linspace(5, 5, 170)
landing_temperature = np.concatenate((landing_temperature1, landing_temperature2))

# 模拟降落阶段功率变化
landing_time1 = np.linspace(400, 450, 50)
landing_time2 = np.linspace(450, 500, 150)
landing_power = np.concatenate(
    ((landing_time1 - 400) * 30 + 1000, landing_time2 * 0 + 2500))

# 合并时间、温度、功率数据
time = np.concatenate((takeoff_time, cruise_time, landing_time))
temperature = np.concatenate((takeoff_temperature, cruise_temperature, landing_temperature))
power = np.concatenate((takeoff_power, cruise_power, landing_power))




def plot_base_Curve():

    np.random.seed(1)
    # 定义滤波窗口大小和多项式阶数（可根据实际情况调整）

    # 1. 对温度数据进行滤波处理
    filtered_temperature = savgol_filter(temperature, 100, 3)

    # 2. 对滤波后的温度数据添加噪声，使其更随机
    noise_temperature = np.random.normal(0, TEMPERATURE_NOISE_STD, size=filtered_temperature.shape)  # 均值为0，标准差为3的噪声，可调整
    temperature_with_noise = filtered_temperature + noise_temperature

    # 3. 对功率数据进行滤波处理
    filtered_power = savgol_filter(power, 100, 5)

    # 4. 对滤波后的功率数据添加噪声，使其更随机
    noise_power = np.random.normal(0, POWER_NOISE_STD, size=filtered_power.shape)  # 均值为0，标准差为100的噪声，可调整
    power_with_noise = filtered_power + noise_power

    # 绘制图形
    fig, ax_temp = plt.subplots(figsize=(8, 6), dpi=100)

    # 绘制温度随时间变化曲线（带噪声）
    ax_temp.set_xlabel('时间 (s)')
    ax_temp.set_ylabel('环境温度 (℃)', color='#1f77b4')
    temperature_line, = ax_temp.plot(time, temperature_with_noise, color='#1f77b4', label='环境温度')
    ax_temp.tick_params(axis='y', labelcolor='#1f77b4')

    # 设置横轴和纵轴显示范围
    plt.xlim(0, 600)
    plt.ylim(-100, 40)

    # 创建次坐标轴用于绘制功率曲线（带噪声）
    ax_power = ax_temp.twinx()
    ax_power.set_ylabel('功率需求 (W)', color='#ff7f0e')
    power_line, = ax_power.plot(time, power_with_noise, color='#ff7f0e', label='功率需求', linestyle='--')
    ax_power.tick_params(axis='y', labelcolor='#ff7f0e')

    # 阶段背景绘制
    ax_temp.axvspan(0, 200, alpha=0.2, color='lightblue', label='飞行阶段')
    ax_temp.axvspan(200, 400, alpha=0.2, color='lightgreen', label='水面滑行')
    ax_temp.axvspan(400, 600, alpha=0.2, color='salmon', label='水下潜航')

    # 巡航阶段透明框
    rect_temperature = mpatches.Rectangle((200, 15), 200, 20,
                                          linewidth=1, edgecolor='none',
                                          facecolor='gray', alpha=0.2,
                                          transform=ax_temp.transData)
    ax_temp.add_patch(rect_temperature)

    rect_power = mpatches.Rectangle((200, 800), 200, 400,
                                    linewidth=1, edgecolor='none',
                                    facecolor='gray', alpha=0.2,
                                    transform=ax_power.transData)
    ax_power.add_patch(rect_power)

    # 添加网格线
    ax_temp.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 图例设置
    taking_off_patch = mpatches.Patch(color='lightblue', label='飞行阶段', alpha=0.2)
    cruising_patch = mpatches.Patch(color='lightgreen', label='水面滑行', alpha=0.2)
    landing_patch = mpatches.Patch(color='salmon', label='水下潜航', alpha=0.2)
    ax_temp.legend(handles=[power_line, temperature_line, taking_off_patch, cruising_patch, landing_patch],
                   fontsize=14, loc="lower right",
                   frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

    # 箭头和注释
    arrow_style = dict(arrowstyle="<->", facecolor='#16344C', edgecolor='#16344C', lw=2)

    ax_temp.annotate('', xy=(cruise_time[50], cruise_temperature[50] - 10),
                     xytext=(cruise_time[50], cruise_temperature[50] + 10),
                     arrowprops=arrow_style)

    ax_power.annotate('', xy=(cruise_time[50], cruise_power[50] - 200),
                      xytext=(cruise_time[50], cruise_power[50] + 200),
                      arrowprops=arrow_style)

    ax_temp.annotate('25 ± 10℃', xy=(cruise_time[60], cruise_temperature[52] - 15), fontsize=16)
    ax_power.annotate('1000±200W', xy=(cruise_time[60], cruise_power[53] + 250), fontsize=16)

    fig.savefig('../Figures/Power_Temperature.svg')
    plt.show()


class UAV_Load(object):
    def __init__(self):
        pass

    @staticmethod
    def get_loads(plot=False):
        """返回带高斯噪声的环境数据"""
        return temperature, power


if __name__ == '__main__':
    plot_base_Curve()