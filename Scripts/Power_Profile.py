# Power_Profile.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = ('Times New Roman')
plt.rcParams['font.size'] = 12

# 默认噪声参数与滤波参数（可在函数调用时覆盖）
TEMPERATURE_NOISE_STD = 3.0  # ℃
POWER_NOISE_STD = 80.0  # W

# savgol 滤波要求 window_length 为奇数且 <= len(data)
DEFAULT_TEMP_SAVGOL_WINDOW = 101
DEFAULT_TEMP_SAVGOL_POLY = 3

DEFAULT_POWER_SAVGOL_WINDOW = 101
DEFAULT_POWER_SAVGOL_POLY = 5


def _build_base_profiles():
    """
    构建基础（无噪声）时间序列、温度与功率剖面。
    返回：time, temperature, power （np.ndarray）
    """
    # 起飞阶段 (0 - 200)
    takeoff_time = np.linspace(0, 200, 200)
    takeoff_temperature1 = np.linspace(-75, -75, 150)
    takeoff_temperature2 = np.linspace(-75, 25, 50)
    takeoff_temperature = np.concatenate((takeoff_temperature1, takeoff_temperature2))

    # 起飞功率
    takeoff_time1 = np.linspace(0, 150, 150)
    takeoff_time2 = np.linspace(150, 170, 20)
    takeoff_time3 = np.linspace(170, 200, 30)
    takeoff_power = np.concatenate(
        (takeoff_time1 * 0 + 4000, takeoff_time2 * 40 - 2000, (takeoff_time3 - 170) * (-120) + 4600)
    )

    # 巡航阶段 (200 - 400)
    cruise_time = np.linspace(200, 400, 200)
    cruise_temperature = np.full(200, 25)
    cruise_power = np.full(200, 1000)

    # 降落阶段 (400 - 600)
    landing_time = np.linspace(400, 600, 200)
    landing_temperature1 = np.linspace(25, 5, 30)
    landing_temperature2 = np.linspace(5, 5, 170)
    landing_temperature = np.concatenate((landing_temperature1, landing_temperature2))

    landing_time1 = np.linspace(400, 450, 50)
    landing_time2 = np.linspace(450, 500, 150)
    landing_power = np.concatenate(
        ((landing_time1 - 400) * 30 + 1000, landing_time2 * 0 + 2500)
    )

    time = np.concatenate((takeoff_time, cruise_time, landing_time))
    temperature = np.concatenate((takeoff_temperature, cruise_temperature, landing_temperature))
    power = np.concatenate((takeoff_power, cruise_power, landing_power))

    return time, temperature, power


def generate_loads(seed: int | None = None,
                   temp_noise_std: float = TEMPERATURE_NOISE_STD,
                   power_noise_std: float = POWER_NOISE_STD,
                   temp_savgol_window: int = DEFAULT_TEMP_SAVGOL_WINDOW,
                   temp_savgol_poly: int = DEFAULT_TEMP_SAVGOL_POLY,
                   power_savgol_window: int = DEFAULT_POWER_SAVGOL_WINDOW,
                   power_savgol_poly: int = DEFAULT_POWER_SAVGOL_POLY):
    """
    生成带噪声且经滤波的温度与功率需求序列（np.ndarray）。
    返回 (temperature_with_noise, power_with_noise, time)

    参数:
        seed: 随机种子（None 则不设置，便于外部随机）
        temp_noise_std: 温度噪声标准差 (℃)
        power_noise_std: 功率噪声标准差 (W)
        *_savgol_*: savgol_filter 的窗口与多项式阶数（window 必须为奇数）
    """
    # 构建基础剖面
    time, temperature_base, power_base = _build_base_profiles()

    # 随机种子设置（非 None 时确定性）
    rnd = np.random.RandomState(seed) if seed is not None else np.random

    # 调整窗口（确保为奇数并且 < len）
    def _ensure_odd_window(win, length):
        win = int(win)
        if win >= length:
            win = length - 1
        if win % 2 == 0:
            win = max(3, win - 1)
        return win

    temp_win = _ensure_odd_window(temp_savgol_window, len(temperature_base))
    power_win = _ensure_odd_window(power_savgol_window, len(power_base))

    # 1) 对温度数据进行 savgol 滤波（去趋势）
    try:
        filtered_temperature = savgol_filter(temperature_base, temp_win, temp_savgol_poly)
    except Exception:
        filtered_temperature = temperature_base.copy()

    # 2) 添加高斯噪声
    noise_temperature = rnd.normal(0.0, temp_noise_std, size=filtered_temperature.shape)
    temperature_with_noise = filtered_temperature + noise_temperature

    # 3) 对功率数据滤波
    try:
        filtered_power = savgol_filter(power_base, power_win, power_savgol_poly)
    except Exception:
        filtered_power = power_base.copy()

    # 4) 添加高斯噪声
    noise_power = rnd.normal(0.0, power_noise_std, size=filtered_power.shape)
    power_with_noise = filtered_power + noise_power

    # 确保返回为 np.ndarray（float 类型）
    return np.asarray(temperature_with_noise, dtype=np.float32), np.asarray(power_with_noise, dtype=np.float32), np.asarray(time, dtype=np.float32)


def plot_base_curve(time: np.ndarray,
                    temperature: np.ndarray,
                    power: np.ndarray,
                    save_path: str | None = "../Figures/Power_Temperature.svg",
                    show: bool = True):
    """
    绘制温度-功率曲线，风格与原脚本保持一致。

    参数:
        time, temperature, power: np.ndarray
        save_path: 若不为 None，则保存为该文件
        show: 是否调用 plt.show()
    """
    fig, ax_temp = plt.subplots(figsize=(8, 6), dpi=100)

    # 温度曲线
    ax_temp.set_xlabel('时间 (s)')
    ax_temp.set_ylabel('环境温度 (°C)', color='#1f77b4')
    temperature_line, = ax_temp.plot(time, temperature, color='#1f77b4', label='环境温度')
    ax_temp.tick_params(axis='y', labelcolor='#1f77b4')

    # 设置 x/y 范围
    ax_temp.set_xlim(0, 600)
    ax_temp.set_ylim(-100, 40)

    # 功率曲线（用第二个 y 轴）
    ax_power = ax_temp.twinx()
    ax_power.set_ylabel('功率需求 (W)', color='#ff7f0e')
    power_line, = ax_power.plot(time, power, color='#ff7f0e', label='功率需求', linestyle='--')
    ax_power.tick_params(axis='y', labelcolor='#ff7f0e')

    # 阶段背景
    ax_temp.axvspan(0, 200, alpha=0.2, color='lightblue', label='飞行阶段')
    ax_temp.axvspan(200, 400, alpha=0.2, color='lightgreen', label='水面滑行')
    ax_temp.axvspan(400, 600, alpha=0.2, color='salmon', label='水下潜航')

    # 巡航阶段透明框（视觉辅助）
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

    # 网格与图例
    ax_temp.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)

    taking_off_patch = mpatches.Patch(color='lightblue', label='飞行阶段', alpha=0.2)
    cruising_patch = mpatches.Patch(color='lightgreen', label='水面滑行', alpha=0.2)
    landing_patch = mpatches.Patch(color='salmon', label='水下潜航', alpha=0.2)

    # 将两条曲线与阶段图例合并显示
    ax_temp.legend(handles=[power_line, temperature_line, taking_off_patch, cruising_patch, landing_patch],
                   fontsize=14, loc="lower right",
                   frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

    # 注释与箭头（保留原有风格）
    arrow_style = dict(arrowstyle="<->", facecolor='#16344C', edgecolor='#16344C', lw=2)
    # 选择巡航阶段中点的一些索引用于注释（防止越界）
    mid_idx = min(len(time) - 1, max(0, int(len(time) * 0.5)))
    try:
        ax_temp.annotate('', xy=(time[mid_idx], 15), xytext=(time[mid_idx], 15 + 20), arrowprops=arrow_style)
        ax_power.annotate('', xy=(time[mid_idx], 1000 - 200), xytext=(time[mid_idx], 1000 + 200), arrowprops=arrow_style)
        ax_temp.annotate('25 ± 10°C', xy=(time[mid_idx], 5), fontsize=16)
        ax_power.annotate('1000±200W', xy=(time[mid_idx], 1250), fontsize=16)
    except Exception:
        pass

    # 保存与显示
    if save_path is not None:
        try:
            fig.savefig(save_path)
        except Exception:
            pass

    if show:
        plt.show()
    else:
        plt.close(fig)


class UAV_Load(object):
    """
    提供静态方法 get_loads(plot=False, seed=None)
    - 返回： (temperature_np, power_np)
    """
    @staticmethod
    def get_loads(plot: bool = False, seed: int | None = None):
        """
        生成并返回带噪声与滤波后的温度与功率数组（numpy）。
        如果 plot=True 会绘制图形。
        """
        temperature, power, time = generate_loads(seed=seed)
        if plot:
            plot_base_curve(time, temperature, power)
        # 与旧接口兼容：返回 (temperature, power)
        return temperature, power


# 当作为脚本直接运行时，绘制图像
if __name__ == '__main__':
    temps, pows, t = generate_loads(seed=1)
    plot_base_curve(t, temps, pows, save_path="../Figures/Power_Temperature.svg", show=True)
