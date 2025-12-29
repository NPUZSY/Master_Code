import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter


# 模拟字体设置（与Power_Profile.py一致）
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()
global_cmap = "coolwarm"

np.random.seed(0)
plt.ion()
SAVE_FIGURES = True


# -------------------------- 第一步：提取功率数据 --------------------------
def _build_base_profiles():
    """从Power_Profile.py中复制的功率生成逻辑，用于获取功率序列"""
    # 起飞阶段 (0 - 200)
    takeoff_time = np.linspace(0, 200, 200)
    takeoff_time1 = np.linspace(0, 150, 150)
    takeoff_time2 = np.linspace(150, 170, 20)
    takeoff_time3 = np.linspace(170, 200, 30)
    takeoff_power = np.concatenate(
        (takeoff_time1 * 0 + 2000,
         takeoff_time2 * 20 - 1000,
         (takeoff_time3 - 170) * (-60) + 3000)
    )

    # 巡航阶段 (200 - 400)
    cruise_time = np.linspace(200, 400, 200)
    cruise_power = np.full(200, 1000)

    # 降落阶段 (400 - 600)
    landing_time = np.linspace(400, 600, 200)
    landing_time1 = np.linspace(400, 450, 50)
    landing_time2 = np.linspace(450, 500, 150)
    landing_power = np.concatenate(
        ((landing_time1 - 400) * 30 + 1000, landing_time2 * 0 + 2500)
    )

    # 再出水飞行阶段 (600 - 800)
    reflight_time = np.linspace(600, 800, 200)
    reflight_power1 = np.linspace(2000, 5000, 5)
    reflight_power2 = np.linspace(5000, 2000, 45)
    reflight_power3 = np.full(150, 2000)
    reflight_power = np.concatenate((reflight_power1, reflight_power2, reflight_power3))

    # 合并所有阶段功率数据
    power = np.concatenate((takeoff_power, cruise_power, landing_power, reflight_power))
    return power


def data_read_process():
    """替代原温度数据处理，改用功率数据"""
    # 获取功率序列
    power = _build_base_profiles()
    # （可选）添加噪声（与Power_Profile.py的generate_loads逻辑一致）
    rnd = np.random.RandomState(seed=1)
    power_noise_std = 80.0
    power_with_noise = power + rnd.normal(0.0, power_noise_std, size=power.shape)
    # 滤波（可选）
    try:
        power_win = 11 if len(power_with_noise)>=11 else len(power_with_noise)-1
        if power_win%2 ==0: power_win -=1
        filtered_power = savgol_filter(power_with_noise, power_win, 5)
    except:
        filtered_power = power_with_noise.copy()

    # 功率数据处理（类比原温度逻辑）
    min_power = min(filtered_power)
    max_power = max(filtered_power)
    power_interval = 100  # 功率区间步长（可根据需求调整）
    power_interval_sample_interval = 10  # 采样间隔（类比原temp_interval_sample_interval）
    num_intervals = int((max_power - min_power) / power_interval) + 1

    temp_counts = np.zeros(num_intervals)
    probability_matrix = np.zeros((num_intervals, num_intervals, 1))
    for i in range(len(filtered_power) - power_interval_sample_interval):
        current_power = filtered_power[i]
        next_power = filtered_power[i + power_interval_sample_interval]
        if np.isnan(current_power) or np.isnan(next_power):
            continue
        current_idx = int((current_power - min_power) / power_interval)
        next_idx = int((next_power - min_power) / power_interval)
        # 防止索引越界
        current_idx = np.clip(current_idx, 0, num_intervals-1)
        next_idx = np.clip(next_idx, 0, num_intervals-1)
        probability_matrix[current_idx, next_idx, 0] += 1
        temp_counts[current_idx] += 1
    
    power_probabilities = temp_counts / np.sum(temp_counts) if np.sum(temp_counts)!=0 else temp_counts
    return probability_matrix, min_power, max_power, num_intervals, power_interval, power_probabilities


# -------------------------- 以下函数与原代码逻辑一致（仅适配变量名） --------------------------
def normalize(probability_matrix):
    probability_matrix_normalize = probability_matrix.copy()
    row_sums = np.sum(probability_matrix_normalize, axis=1)
    for i in range(probability_matrix_normalize.shape[0]):
        if row_sums[i] > 0:
            probability_matrix_normalize[i] /= row_sums[i]
    return probability_matrix_normalize


def interpolation(probability_matrix, min_power, max_power, interpolation_factor=2):
    num_intervals_original = probability_matrix.shape[0]
    num_intervals_interpolated = num_intervals_original * interpolation_factor

    power_interval_original = (max_power - min_power) / num_intervals_original
    power_interval_new = power_interval_original / interpolation_factor

    x_axis_labels_interpolated = [min_power + i * (power_interval_original / interpolation_factor) 
                                  for i in range(num_intervals_interpolated)]
    y_axis_labels_interpolated = [min_power + i * (power_interval_original / interpolation_factor) 
                                  for i in range(num_intervals_interpolated)]

    x_grid = np.linspace(min_power, max_power, num_intervals_original)
    y_grid = np.linspace(min_power, max_power, num_intervals_original)
    points = (x_grid, y_grid)

    interpolate_func = RegularGridInterpolator(points, probability_matrix[:, :, 0], method='linear', 
                                               bounds_error=False, fill_value=None)

    xi = np.array([(x, y) for x in x_axis_labels_interpolated for y in y_axis_labels_interpolated])
    probability_matrix_interpolated = interpolate_func(xi).reshape(num_intervals_interpolated,
                                                                   num_intervals_interpolated, 1)

    return probability_matrix_interpolated, min_power, num_intervals_interpolated, power_interval_new


def plot_surf(probability_matrix, min_power, max_power, num_intervals, power_interval, power_probabilities, z_label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_axis_labels = [min_power + i * power_interval for i in range(num_intervals)]
    y_axis_labels = [min_power + i * power_interval for i in range(num_intervals)]
    X, Y = np.meshgrid(x_axis_labels, y_axis_labels)
    Z = probability_matrix[:, :, 0]

    surf = ax.plot_surface(X, Y, Z, cmap=global_cmap, alpha=0.8)
    ax.view_init(elev=30, azim=250)
    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.125,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )
    cax = fig.add_axes((0.8, 0.25, 0.02, 0.5))
    fig.colorbar(surf, cax=cax, shrink=0.5, aspect=5)

    ax.set_xlabel('Current Power/W')
    ax.set_ylabel('Next Power/W')
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title)
    # 绘制功率出现概率曲线
    x_curve = np.arange(num_intervals) * power_interval + min_power
    z_curve = power_probabilities
    ax.plot(x_curve, [max_power] * len(x_curve), z_curve, '#FFA500', label='Power Occurrence Probability')
    ax.legend(
        fontsize=10,          # 缩小字体（原16→10，可根据需求调整为8/12）
        loc='upper right',    # 基础位置
        bbox_to_anchor=(1.2, 1),  # 往右（x+0.1）、往上（y+0.05）挪动
        frameon=True,
        borderaxespad=0.1,
        framealpha=0.8        # 增加图例透明度，避免遮挡
    )

    # 窗口最大化
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except AttributeError:
        try:
            manager.resize(*manager.window.maxsize())
        except:
            manager.resize(1600, 900)
    
    if SAVE_FIGURES: 
        os.makedirs("./Figures", exist_ok=True)
        plt.savefig(f"./Figures/Fig5-4 {title}_surf.svg")
    plt.ioff()


if __name__ == '__main__':
    # 处理功率数据
    probability_matrix_, min_power_, max_power_, num_intervals, power_interval, power_probabilities_ = data_read_process()
    
    # 高斯滤波
    probability_matrix_ = gaussian_filter(probability_matrix_[:, :, 0], sigma=2)[:, :, np.newaxis]
    
    # 插值
    probability_matrix_, min_power_, num_intervals_interpolated, power_interval_new = interpolation(
        probability_matrix_, min_power_, max_power_, 1
    )
    
    # 归一化
    probability_matrix_ = normalize(probability_matrix_)
    
    # 绘制功率转移曲面图
    plot_surf(probability_matrix_, min_power_, max_power_, num_intervals_interpolated, power_interval_new,
              power_probabilities_, 'Transition Probability', "Power Demand Transition")