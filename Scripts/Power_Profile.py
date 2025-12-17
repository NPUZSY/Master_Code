# Power_Profile.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import savgol_filter
import os
import sys

# 获取当前脚本的绝对路径（无论从哪执行，都能定位到脚本所在目录）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 将脚本所在目录加入Python搜索路径
sys.path.append(current_script_dir)
from utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# 默认噪声参数与滤波参数（可在函数调用时覆盖）
TEMPERATURE_NOISE_STD = 3.0  # ℃
POWER_NOISE_STD = 80.0  # W

# savgol 滤波要求 window_length 为奇数且 <= len(data)
DEFAULT_TEMP_SAVGOL_WINDOW = 101
DEFAULT_TEMP_SAVGOL_POLY = 3

DEFAULT_POWER_SAVGOL_WINDOW = 11
DEFAULT_POWER_SAVGOL_POLY = 5


def _build_base_profiles():
    """
    构建基础（无噪声）时间序列、温度与功率剖面。
    新增600-800秒再出水飞行阶段，参考起飞阶段特征
    返回：time, temperature, power （np.ndarray）
    """
    # 起飞阶段 (0 - 200) - 温度以-15℃为中心，功率2000W基准
    takeoff_time = np.linspace(0, 200, 200)
    # 第一阶段温度修改：前150秒保持-15℃，后50秒上升到25℃（以-15为中心）
    takeoff_temperature1 = np.linspace(-15, -15, 150)
    takeoff_temperature2 = np.linspace(-15, 25, 50)
    takeoff_temperature = np.concatenate((takeoff_temperature1, takeoff_temperature2))

    # 起飞功率 - 保持2000W基准
    takeoff_time1 = np.linspace(0, 150, 150)
    takeoff_time2 = np.linspace(150, 170, 20)
    takeoff_time3 = np.linspace(170, 200, 30)
    takeoff_power = np.concatenate(
        (takeoff_time1 * 0 + 2000,  # 基准功率改为2000W
         takeoff_time2 * 20 - 1000,  # 调整系数使功率合理上升（峰值约3000W）
         (takeoff_time3 - 170) * (-60) + 3000)  # 功率下降段
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

    # 新增：再出水飞行阶段 (600 - 800)
    # 第四阶段温度修改：以20℃为中心，600-750秒从5℃上升到20℃，750-800秒保持20℃
    reflight_time = np.linspace(600, 800, 200)
    # 计算750秒对应的索引：600-800共200个点，750秒是第150个点（600+150=750）
    reflight_temperature1 = np.linspace(5, 20, 50)  # 600-750秒：5→20℃
    reflight_temperature2 = np.linspace(20, 20, 150)   # 750-800秒：保持20℃
    reflight_temperature = np.concatenate((reflight_temperature1, reflight_temperature2))

    # 功率保持原有逻辑：650秒达5000W峰值，700秒下降到2000W，700-800秒保持2000W
    reflight_power1 = np.linspace(2000, 5000, 5)    # 600-650秒：上升至峰值
    reflight_power2 = np.linspace(5000, 2000, 45)    # 650-700秒：快速下降至2000W
    reflight_power3 = np.full(150, 2000)             # 700-800秒：稳定在2000W
    reflight_power = np.concatenate((reflight_power1, reflight_power2, reflight_power3))

    # 合并所有阶段数据
    time = np.concatenate((takeoff_time, cruise_time, landing_time, reflight_time))
    temperature = np.concatenate((takeoff_temperature, cruise_temperature, landing_temperature, reflight_temperature))
    power = np.concatenate((takeoff_power, cruise_power, landing_power, reflight_power))

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
    """
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

    # 温度滤波+加噪声
    try:
        filtered_temperature = savgol_filter(temperature_base, temp_win, temp_savgol_poly)
    except Exception:
        filtered_temperature = temperature_base.copy()
    noise_temperature = rnd.normal(0.0, temp_noise_std, size=filtered_temperature.shape)
    temperature_with_noise = filtered_temperature + noise_temperature

    # 功率滤波+加噪声
    try:
        filtered_power = savgol_filter(power_base, power_win, power_savgol_poly)
    except Exception:
        filtered_power = power_base.copy()
    noise_power = rnd.normal(0.0, power_noise_std, size=filtered_power.shape)
    power_with_noise = filtered_power + noise_power

    return np.asarray(temperature_with_noise, dtype=np.float32), np.asarray(power_with_noise, dtype=np.float32), np.asarray(time, dtype=np.float32)


def plot_base_curve(time: np.ndarray,
                    temperature: np.ndarray,
                    power: np.ndarray,
                    save_path: str | None = "../Figures/Power_Temperature.svg",
                    show: bool = True):
    """
    绘制温度-功率曲线，新增再出水飞行阶段的可视化
    """
    fig, ax_temp = plt.subplots(figsize=(10, 6), dpi=100)

    # 温度曲线
    ax_temp.set_xlabel('时间 (s)', fontsize=12)
    ax_temp.set_ylabel('环境温度 (°C)', color='#1f77b4', fontsize=12)
    temperature_line, = ax_temp.plot(time, temperature, color='#1f77b4', label='环境温度')
    ax_temp.tick_params(axis='y', labelcolor='#1f77b4')

    # 温度轴范围修改：-25~40℃
    ax_temp.set_xlim(0, 800)
    ax_temp.set_ylim(-25, 40)

    # 功率曲线（用第二个 y 轴）
    ax_power = ax_temp.twinx()
    ax_power.set_ylabel('功率需求 (W)', color='#ff7f0e', fontsize=12)
    power_line, = ax_power.plot(time, power, color='#ff7f0e', label='功率需求', linestyle='--')
    ax_power.tick_params(axis='y', labelcolor='#ff7f0e')
    ax_power.set_ylim(0, 5500)

    # 阶段背景
    ax_temp.axvspan(0, 200, alpha=0.2, color='lightblue', label='飞行阶段')
    ax_temp.axvspan(200, 400, alpha=0.2, color='lightgreen', label='水面滑行')
    ax_temp.axvspan(400, 600, alpha=0.2, color='salmon', label='水下潜航')
    ax_temp.axvspan(600, 800, alpha=0.2, color='mediumpurple', label='再出水飞行')

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

    # 阶段图例
    taking_off_patch = mpatches.Patch(color='lightblue', label='高空飞行', alpha=0.2)
    cruising_patch = mpatches.Patch(color='lightgreen', label='水面滑行', alpha=0.2)
    landing_patch = mpatches.Patch(color='salmon', label='水下潜航', alpha=0.2)
    reflight_patch = mpatches.Patch(color='mediumpurple', label='出水飞行', alpha=0.2)

    ax_temp.legend(handles=[power_line, temperature_line, taking_off_patch, cruising_patch, 
                           landing_patch, reflight_patch],
                   fontsize=12, loc="lower right",
                   frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

    # 注释与箭头
    arrow_style = dict(arrowstyle="<->", facecolor='#16344C', edgecolor='#16344C', lw=2)
    mid_idx = min(len(time) - 1, max(0, int(len(time) * 0.375)))  # 巡航阶段中点
    try:
        # 巡航阶段温度/功率注释
        ax_temp.annotate('', xy=(time[mid_idx], 15), xytext=(time[mid_idx], 15 + 20), arrowprops=arrow_style)
        ax_power.annotate('', xy=(time[mid_idx], 1000 - 200), xytext=(time[mid_idx], 1000 + 200), arrowprops=arrow_style)
        ax_temp.annotate('25 ± 10°C', xy=(time[mid_idx], 10), fontsize=14)
        ax_power.annotate('1000±200W', xy=(time[mid_idx], 1250), fontsize=14)
    except Exception as e:
        print(f"添加注释失败: {e}")

    # 保存与显示
    if save_path is not None:
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=1200)
        except Exception as e:
            print(f"保存图片失败: {e}")

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
        return temperature, power


# 当作为脚本直接运行时，绘制图像
if __name__ == '__main__':
    # 确保创建保存目录
    import os
    save_dir = "./Figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    temps, pows, t = generate_loads(seed=1)
    plot_base_curve(t, temps, pows, save_path="./Figures/Power_Temperature.svg", show=True)
    plot_base_curve(t, temps, pows, save_path="./Figures/Power_Temperature.png")