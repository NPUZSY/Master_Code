import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，确保在没有图形界面的环境中也能运行
import matplotlib.pyplot as plt
import os
import sys

# 模拟字体设置（与Power_Profile.py一致）
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# 设置字体为Times New Roman（确保图表字体规范）
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 创建Figures目录（若不存在则自动创建）
os.makedirs('./Figures', exist_ok=True)

# 定义基础功率值（仅用于示例展示，核心体现P1到P2的切换逻辑）
P_air = 2500  # 空中飞行功率（W）
P_water_surface = 1000  # 水面航行功率（W）
P_underwater = 3000  # 水下潜航功率（W）

# 定义各切换模态的功率切换函数（修复时间区间间隙+异常功率）
def air_to_surface(t, P1=P_air, P2=P_water_surface):
    """空中-水面切换模态：P1→1.1P1→0.9P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 20:
            # 过渡中阶段1：P1→1.1P1
            ratio = (ti - 10) / 10  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.1 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：1.1P1→0.9P2
            ratio = (ti - 20) / 15  # 归一化到[0,1]
            power[i] = 1.1 * P1 + ratio * (0.9 * P2 - 1.1 * P1)
        elif ti <= 40:
            # 过渡中阶段3：0.9P2→P2
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 0.9 * P2 + ratio * (P2 - 0.9 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

def surface_to_air(t, P1=P_water_surface, P2=P_air):
    """水面-空中切换模态：P1→0.9P1→1.05P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 20:
            # 过渡中阶段1：P1→0.9P1
            ratio = (ti - 10) / 10  # 归一化到[0,1]
            power[i] = P1 + ratio * (0.9 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：0.9P1→1.05P2
            ratio = (ti - 20) / 15  # 归一化到[0,1]
            power[i] = 0.9 * P1 + ratio * (1.05 * P2 - 0.9 * P1)
        elif ti <= 40:
            # 过渡中阶段3：1.05P2→P2
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 1.05 * P2 + ratio * (P2 - 1.05 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

def air_to_underwater(t, P1=P_air, P2=P_underwater):
    """空中-水下切换模态：P1→P1→1.2P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：维持P1
            power[i] = P1
        elif ti <= 30:
            # 过渡中阶段2：P1→1.2P2
            ratio = (ti - 25) / 5  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.2 * P2 - P1)
        elif ti <= 40:
            # 过渡中阶段3：1.2P2→P2
            ratio = (ti - 30) / 10  # 归一化到[0,1]
            power[i] = 1.2 * P2 + ratio * (P2 - 1.2 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

def underwater_to_air(t, P1=P_underwater, P2=P_air):
    """水下-空中切换模态：P1→1.1P1→P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 20:
            # 过渡中阶段1：P1→1.1P1
            ratio = (ti - 10) / 10  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.1 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：1.1P1→P2
            ratio = (ti - 20) / 15  # 归一化到[0,1]
            power[i] = 1.1 * P1 + ratio * (P2 - 1.1 * P1)
        elif ti <= 40:
            # 过渡中阶段3：维持P2
            power[i] = P2
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

def surface_to_underwater(t, P1=P_water_surface, P2=P_underwater):
    """水面-水下切换模态：P1→1.05P1→P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：P1→1.05P1
            ratio = (ti - 10) / 15  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.05 * P1 - P1)
        elif ti <= 40:
            # 过渡中阶段2：1.05P1→P2
            ratio = (ti - 25) / 15  # 归一化到[0,1]
            power[i] = 1.05 * P1 + ratio * (P2 - 1.05 * P1)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

def underwater_to_surface(t, P1=P_underwater, P2=P_water_surface):
    """水下-水面切换模态：P1→0.95P1→0.9P2→P2"""
    power = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：P1→0.95P1
            ratio = (ti - 10) / 15  # 归一化到[0,1]
            power[i] = P1 + ratio * (0.95 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：0.95P1→0.9P2
            ratio = (ti - 25) / 10  # 归一化到[0,1]
            power[i] = 0.95 * P1 + ratio * (0.9 * P2 - 0.95 * P1)
        elif ti <= 40:
            # 过渡中阶段3：0.9P2→P2
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 0.9 * P2 + ratio * (P2 - 0.9 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    return power

# 生成时间轴（0-50s，间隔0.1s，确保曲线平滑）
time = np.linspace(0, 50, 501)

# 创建图表（尺寸适配论文插图规范）
fig, ax = plt.subplots(figsize=(12, 5))

# 定义各模态的颜色和标签（保持视觉区分度）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
modes = [
    (air_to_surface, 'Air to Surface'),
    (surface_to_air, 'Surface to Air'),
    (air_to_underwater, 'Air to Underwater'),
    (underwater_to_air, 'Underwater to Air'),
    (surface_to_underwater, 'Surface to Underwater'),
    (underwater_to_surface, 'Underwater to Surface')
]

# 绘制各模态的功率切换曲线
for i, (mode_func, mode_name) in enumerate(modes):
    power = mode_func(time)
    ax.plot(time, power, color=colors[i], label=mode_name, linewidth=2.5, alpha=0.8)

# 添加阶段划分背景色（直观区分三个阶段）
ax.axvspan(0, 10, alpha=0.1, color='gray', label='Pre-transition')
ax.axvspan(10, 40, alpha=0.2, color='yellow', label='Transition')
ax.axvspan(40, 50, alpha=0.1, color='green', label='Post-transition')

# 添加基础功率参考线（便于对比初始/目标功率）
ax.axhline(y=P_air, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Air Power ({P_air}W)')
ax.axhline(y=P_water_surface, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Surface Power ({P_water_surface}W)')
ax.axhline(y=P_underwater, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Underwater Power ({P_underwater}W)')

# 设置图表属性（符合论文格式要求）
ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Power (W)', fontsize=14, fontweight='bold')
ax.set_title('Power Switching Process of Different Modes', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(0, 50)
ax.set_ylim(0, 4000)  # 适配最大功率显示

# 调整图例位置（避免遮挡曲线）
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9, shadow=True)

# 添加阶段标注（增强图表可读性）
ax.text(5, ax.get_ylim()[1] * 0.9, 'Pre-transition\n(0~10s)', ha='center', fontsize=11, 
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
ax.text(25, ax.get_ylim()[1] * 0.9, 'Transition\n(11~40s)', ha='center', fontsize=11,
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
ax.text(45, ax.get_ylim()[1] * 0.9, 'Post-transition\n(41~50s)', ha='center', fontsize=11,
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

# 美化图表边框（隐藏上、右边框，增强简洁性）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)

# 调整布局（预留图例空间）
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 保存为SVG格式（高清无失真，适配论文排版）
plt.savefig('./Figures/Fig5-6 Power Switching Modes Design.svg', format='svg', dpi=300, bbox_inches='tight')

# 关闭图表释放资源
plt.close()

print("图表已保存至 ./Figures/Fig5-6 Power Switching Modes Design.svg")