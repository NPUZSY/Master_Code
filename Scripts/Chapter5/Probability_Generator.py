import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm  # 明确从matplotlib.cm模块导入get_cmap函数
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


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


def data_read_process():
    data = np.load("./Data/combined_icartt_data.npy")

    # 假设数据存储在data数组中，第一列是时间，第二列是温度
    # 数据清洗和转换部分（此处简单示例，可根据实际情况完善）
    data[:, 1] = ((data[:, 1] - 273150) // 1000)  # 假设温度数据从开尔文转换为摄氏度
    # ========== 新增：温度范围缩放（-80~20 → -20~25） ==========
    # 原始温度范围
    old_min = -80
    old_max = 20
    # 目标温度范围
    new_min = -25
    new_max = 25
    # 线性缩放公式
    data[:, 1] = ((data[:, 1] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    # 可选：限制边界值（防止浮点误差超出范围）
    data[:, 1] = np.clip(data[:, 1], new_min, new_max)
    # ==========================================================

    min_temp = min(data[:, 1])
    max_temp = max(data[:, 1])
    temp_interval = 1
    temp_interval_sample_interval = 100
    num_intervals = int((max_temp - min_temp) / temp_interval) + 1
    temp_counts = np.zeros(num_intervals)
    probability_matrix = np.zeros((num_intervals, num_intervals, 1))
    for i in range(len(data) - temp_interval_sample_interval):
        current_temp = data[i, 1]

        next_temp = data[i + temp_interval_sample_interval, 1]
        if np.isnan(current_temp) or np.isnan(next_temp):
            continue
        current_idx = int((current_temp - min_temp) / temp_interval)
        next_idx = int((next_temp - min_temp) / temp_interval)
        probability_matrix[current_idx, next_idx, 0] += 1
        temp_counts[current_idx] += 1
    temp_probabilities = temp_counts / np.sum(temp_counts)
    return probability_matrix, min_temp, max_temp, num_intervals, temp_interval, temp_probabilities


def normalize(probability_matrix):
    probability_matrix_normalize = probability_matrix.copy()
    # 归一化概率矩阵
    row_sums = np.sum(probability_matrix_normalize, axis=1)
    for i in range(probability_matrix_normalize.shape[0]):
        if row_sums[i] > 0:
            probability_matrix_normalize[i] /= row_sums[i]
    return probability_matrix_normalize


def moving_average_smoothing(probability_matrix, window_size=3):
    """
    对二维概率矩阵进行移动平均平滑处理

    参数:
    probability_matrix (numpy.ndarray): 输入的概率矩阵，形状为 (num_intervals, num_intervals, depth)
    window_size (int): 移动平均的窗口大小，默认为3

    返回:
    numpy.ndarray: 平滑后的概率矩阵，形状与输入矩阵相同
    """
    rows, cols, depth = probability_matrix.shape
    smoothed_matrix = np.zeros_like(probability_matrix)
    half_window = window_size // 2
    for k in range(depth):
        for i in range(rows):
            for j in range(cols):
                row_start = max(i - half_window, 0)
                row_end = min(i + half_window + 1, rows)
                col_start = max(j - half_window, 0)
                col_end = min(j + half_window + 1, cols)
                sub_matrix = probability_matrix[row_start:row_end, col_start:col_end, k]
                smoothed_matrix[i, j, k] = np.mean(sub_matrix)
    return smoothed_matrix


def interpolation(probability_matrix, min_temp, max_temp, interpolation_factor=2):
    # 获取原始温度区间数量
    num_intervals_original = probability_matrix.shape[0]

    # 确定新的、更密集的温度区间数量（可根据需要调整这个倍数来控制插值后的密度）
    num_intervals_interpolated = num_intervals_original * interpolation_factor

    temp_interval_original = (max_temp - min_temp) / num_intervals_original
    temp_interval_new = temp_interval_original / interpolation_factor

    # 生成新的、更密集的温度区间坐标（用于插值后的x、y轴坐标）
    x_axis_labels_interpolated = [min_temp + i * (temp_interval_original / interpolation_factor) for i in
                                  range(num_intervals_interpolated)]
    y_axis_labels_interpolated = [min_temp + i * (temp_interval_original / interpolation_factor) for i in
                                  range(num_intervals_interpolated)]

    # 创建用于RegularGridInterpolator的坐标网格（注意维度顺序和范围）
    x_grid = np.linspace(min_temp, max_temp, num_intervals_original)
    y_grid = np.linspace(min_temp, max_temp, num_intervals_original)
    points = (x_grid, y_grid)

    # 创建RegularGridInterpolator对象，用于插值
    interpolate_func = RegularGridInterpolator(points, probability_matrix[:, :, 0], method='linear', bounds_error=False,
                                               fill_value=None)

    # 生成插值点坐标（这里要与RegularGridInterpolator要求的格式匹配）
    xi = np.array([(x, y) for x in x_axis_labels_interpolated for y in y_axis_labels_interpolated])

    # 在新的温度区间坐标上进行插值，得到插值后的概率矩阵
    probability_matrix_interpolated = interpolate_func(xi).reshape(num_intervals_interpolated,
                                                                   num_intervals_interpolated, 1)

    return probability_matrix_interpolated, min_temp, num_intervals_interpolated, temp_interval_new


def plot_bar(probability_matrix, min_temp, num_intervals, temp_interval, z_label, title):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # 确定x、y坐标（每个温度区间的中心值）
    x_axis_labels = [min_temp + (i + 0.5) * temp_interval for i in range(num_intervals)]
    y_axis_labels = [min_temp + (i + 0.5) * temp_interval for i in range(num_intervals)]
    X, Y = np.meshgrid(x_axis_labels, y_axis_labels)

    # 柱子的宽度和深度（根据温度区间大小确定）
    dx = dy = temp_interval
    dz = probability_matrix[:, :, 0]

    # 获取一个合适的颜色映射，这里使用'viridis'，你也可以选择其他如'plasma'、'jet'等
    cmap = cm.get_cmap(global_cmap)

    # 将概率值归一化到0-1区间，以便与颜色映射的范围匹配
    normalized_dz = (dz - np.min(dz)) / (np.max(dz) - np.min(dz))

    # 为每个柱子获取对应的颜色
    colors = cmap(normalized_dz.ravel())

    # 绘制三维柱状图，并设置颜色
    ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz).ravel(), dx, dy, dz.ravel(), color=colors, alpha=0.8)
    ax.view_init(elev=30, azim=250)
    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.125,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )

    ax.set_xlabel('Current Temperature/°C')
    ax.set_ylabel('Next Temperature/°C')
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title)

    # 跨平台最大化窗口适配
    manager = plt.get_current_fig_manager()
    try:
        # Windows (TkAgg/QtAgg)
        manager.window.showMaximized()
    except AttributeError:
        try:
            # Linux (GTK)
            manager.resize(*manager.window.maxsize())
        except:
            # 通用 fallback：设置大尺寸
            manager.resize(1600, 900)
    
    if SAVE_FIGURES: plt.savefig(f"./Figures/Fig5-3 温度转移概率/{title}_bar.png", dpi=1200)
    plt.ioff()


# 生成曲面图
def plot_surf(probability_matrix, min_temp, max_temp, num_intervals, temp_interval, temp_probabilities, z_label, title):
    # 绘制三维图
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    x_axis_labels = [min_temp + i * temp_interval for i in range(num_intervals)]
    y_axis_labels = [min_temp + i * temp_interval for i in range(num_intervals)]
    X, Y = np.meshgrid(x_axis_labels, y_axis_labels)
    Z = probability_matrix[:, :, 0]

    # 使用plot_surface绘制曲面，并设置cmap参数来实现颜色随高度（Z轴数据）变化
    # 这里选用'viridis'颜色映射，你也可以根据喜好选择其他如'plasma'、'jet'等
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
    cax = fig.add_axes((0.8, 0.25, 0.02, 0.5))  # (x坐标, y坐标, 宽度, 高度)

    # 添加颜色条，指定cax参数为新创建的轴对象，同时设置其他参数
    fig.colorbar(surf, cax=cax, shrink=0.5, aspect=5)

    ax.set_xlabel('Current Temperature/°C')
    ax.set_ylabel('Next Temperature/°C')
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title)
    # 绘制表示温度出现概率的曲线
    x_curve = np.arange(num_intervals) * temp_interval + min_temp
    z_curve = temp_probabilities
    ax.plot(x_curve, [max_temp] * len(x_curve), z_curve, '#FFA500', label='Temperature Transition')
    ax.legend(
        fontsize=10,          # 缩小字体（原16→10，可根据需求调整为8/12）
        loc='upper right',    # 基础位置
        bbox_to_anchor=(1.2, 1),  # 往右（x+0.1）、往上（y+0.05）挪动
        frameon=True,
        borderaxespad=0.1,
        framealpha=0.8        # 增加图例透明度，避免遮挡
    )

    # 跨平台最大化窗口适配
    manager = plt.get_current_fig_manager()
    try:
        # Windows (TkAgg/QtAgg)
        manager.window.showMaximized()
    except AttributeError:
        try:
            # Linux (GTK)
            manager.resize(*manager.window.maxsize())
        except:
            # 通用 fallback：设置大尺寸
            manager.resize(1600, 900)
    
    if SAVE_FIGURES: plt.savefig(f"./Figures/Fig5-3 温度转移概率/{title}_surf.png", dpi=1200)
    if SAVE_FIGURES: plt.savefig(f"./Figures/Fig5-3 温度转移概率/{title}_surf.svg", dpi=1200)
    plt.ioff()


if __name__ == '__main__':
    probability_matrix_, min_temp_, max_temp_, num_intervals, temp_interval, temp_probabilities_ = data_read_process()
    plot_bar(probability_matrix_, min_temp_, num_intervals, temp_interval, 'Transition Times', "Source Data")

    # # 移动平均
    # probability_matrix_ = moving_average_smoothing(probability_matrix_, window_size=100)
    # plot_bar(probability_matrix_, min_temp_, num_intervals, temp_interval, 'Transition Times', "Average Data")

    # 高斯滤波
    probability_matrix_ = gaussian_filter(probability_matrix_[:, :, 0], sigma=2)[:, :, np.newaxis]
    plot_bar(probability_matrix_, min_temp_, num_intervals, temp_interval, 'Transition Times', "Filter Data")
    # 插值
    probability_matrix_, min_temp_, num_intervals_interpolated, temp_interval_new = interpolation(probability_matrix_,
                                                                                                  min_temp_, max_temp_,
                                                                                                  1)
    # plot_bar(probability_matrix_, min_temp_, num_intervals_interpolated, temp_interval, 'Transition Times', "Interpolated Data")

    # 归一化
    probability_matrix_ = normalize(probability_matrix_)
    plot_bar(probability_matrix_, min_temp_, num_intervals_interpolated, temp_interval, 'Transition Probability', "Normalized Data")

    plot_surf(probability_matrix_, min_temp_, max_temp_, num_intervals_interpolated, temp_interval_new,
              temp_probabilities_, 'Transition Probability', "Temperature Transition")
    plt.show()