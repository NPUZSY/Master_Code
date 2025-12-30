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

# 定义温度常量
T_air = 0        # 空中温度 (℃)
T_surface = 20   # 水面温度 (℃)
T_underwater = 5 # 水下温度 (℃)

# 定义各切换模态的功率切换函数（修复时间区间间隙+异常功率，取消所有波动）
def air_to_surface(t, P1=P_air, P2=P_water_surface):
    """空中-水面切换模态：P1→1.1P1→0.9P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
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
    # 确保功率非负
    power = np.maximum(power, 0)
    return power

def surface_to_air(t, P1=P_water_surface, P2=P_air):
    """水面-空中切换模态：P1→0.9P1→2.0P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 20:
            # 过渡中阶段1：P1→0.9P1
            ratio = (ti - 10) / 10  # 归一化到[0,1]
            power[i] = P1 + ratio * (0.9 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：0.9P1→2.0P2（修复为2倍P2）
            ratio = (ti - 20) / 15  # 归一化到[0,1]
            power[i] = 0.9 * P1 + ratio * (2.0 * P2 - 0.9 * P1)
        elif ti <= 40:
            # 过渡中阶段3：2.0P2→P2
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 2.0 * P2 + ratio * (P2 - 2.0 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    power = np.maximum(power, 0)
    return power

def air_to_underwater(t, P1=P_air, P2=P_underwater):
    """空中-水下切换模态：P1→P1→1.2P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：维持P1（10-25s）
            power[i] = P1
        elif ti <= 30:
            # 过渡中阶段2：P1→1.2P2（25-30s）
            ratio = (ti - 25) / 5  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.2 * P2 - P1)
        elif ti <= 40:
            # 过渡中阶段3：1.2P2→P2（30-40s）
            ratio = (ti - 30) / 10  # 归一化到[0,1]
            power[i] = 1.2 * P2 + ratio * (P2 - 1.2 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    power = np.maximum(power, 0)
    return power

def underwater_to_air(t, P1=P_underwater, P2=P_air):
    """水下-空中切换模态：P1→1.1P1→2.0P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 20:
            # 过渡中阶段1：P1→1.1P1
            ratio = (ti - 10) / 10  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.1 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：1.1P1→2.0P2（修复为2倍P2）
            ratio = (ti - 20) / 15  # 归一化到[0,1]
            power[i] = 1.1 * P1 + ratio * (2.0 * P2 - 1.1 * P1)
        elif ti <= 40:
            # 过渡中阶段3：2.0P2→P2
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 2.0 * P2 + ratio * (P2 - 2.0 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    power = np.maximum(power, 0)
    return power

def surface_to_underwater(t, P1=P_water_surface, P2=P_underwater):
    """水面-水下切换模态：P1→1.05P1→1.1P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：P1→1.05P1（10-25s）
            ratio = (ti - 10) / 15  # 归一化到[0,1]
            power[i] = P1 + ratio * (1.05 * P1 - P1)
        elif ti <= 40:
            # 过渡中阶段2：1.05P1→1.1P2（25-40s）
            ratio = (ti - 25) / 15  # 归一化到[0,1]
            power[i] = 1.05 * P1 + ratio * (1.1 * P2 - 1.05 * P1)
        else:
            # 过渡后阶段：1.1P2→P2（40s后）
            power[i] = 1.1 * P2 + ((ti - 40)/10) * (P2 - 1.1 * P2)
    power = np.maximum(power, 0)
    return power

def underwater_to_surface(t, P1=P_underwater, P2=P_water_surface):
    """水下-水面切换模态：P1→0.95P1→0.9P2→P2"""
    power = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 10:
            # 过渡前阶段：维持初始功率P1
            power[i] = P1
        elif ti <= 25:
            # 过渡中阶段1：P1→0.95P1（10-25s）
            ratio = (ti - 10) / 15  # 归一化到[0,1]
            power[i] = P1 + ratio * (0.95 * P1 - P1)
        elif ti <= 35:
            # 过渡中阶段2：0.95P1→0.9P2（25-35s）
            ratio = (ti - 25) / 10  # 归一化到[0,1]
            power[i] = 0.95 * P1 + ratio * (0.9 * P2 - 0.95 * P1)
        elif ti <= 40:
            # 过渡中阶段3：0.9P2→P2（35-40s）
            ratio = (ti - 35) / 5  # 归一化到[0,1]
            power[i] = 0.9 * P2 + ratio * (P2 - 0.9 * P2)
        else:
            # 过渡后阶段：维持目标功率P2
            power[i] = P2
    power = np.maximum(power, 0)
    return power

# 定义各模态的温度变化函数
def get_temperature_curve(mode_name, time):
    """生成对应模态的温度变化曲线"""
    temp = np.zeros_like(time, dtype=np.float64)
    for i, ti in enumerate(time):
        if ti <= 10:
            # 前10s：维持初始温度
            if mode_name == 'Air to Surface':
                temp[i] = T_air
            elif mode_name == 'Surface to Air':
                temp[i] = T_surface
            elif mode_name == 'Air to Underwater':
                temp[i] = T_air
            elif mode_name == 'Underwater to Air':
                temp[i] = T_underwater
            elif mode_name == 'Surface to Underwater':
                temp[i] = T_surface
            elif mode_name == 'Underwater to Surface':
                temp[i] = T_underwater
        
        elif 10 < ti <= 40:
            # 过渡阶段（10-40s）：线性变化
            if mode_name == 'Air to Surface':
                # 空中(0℃)→水面(20℃)：线性上升
                ratio = (ti - 10) / 30
                temp[i] = T_air + ratio * (T_surface - T_air)
            
            elif mode_name == 'Surface to Air':
                # 水面(20℃)→空中(0℃)：线性下降
                ratio = (ti - 10) / 30
                temp[i] = T_surface - ratio * (T_surface - T_air)
            
            elif mode_name == 'Air to Underwater':
                # 空中(0℃)→水面(20℃)→水下(5℃)：25s达20℃，先升后降
                if ti <= 25:
                    # 10-25s：空中→水面（0→20℃）
                    ratio = (ti - 10) / 15
                    temp[i] = T_air + ratio * (T_surface - T_air)
                else:
                    # 25-40s：水面→水下（20→5℃）
                    ratio = (ti - 25) / 15
                    temp[i] = T_surface - ratio * (T_surface - T_underwater)
            
            elif mode_name == 'Underwater to Air':
                # 水下(5℃)→水面(20℃)→空中(0℃)：25s达20℃，先升后降
                if ti <= 25:
                    # 10-25s：水下→水面（5→20℃）
                    ratio = (ti - 10) / 15
                    temp[i] = T_underwater + ratio * (T_surface - T_underwater)
                else:
                    # 25-40s：水面→空中（20→0℃）
                    ratio = (ti - 25) / 15
                    temp[i] = T_surface - ratio * (T_surface - T_air)
            
            elif mode_name == 'Surface to Underwater':
                # 水面(20℃)→水下(5℃)：线性下降
                ratio = (ti - 10) / 30
                temp[i] = T_surface - ratio * (T_surface - T_underwater)
            
            elif mode_name == 'Underwater to Surface':
                # 水下(5℃)→水面(20℃)：线性上升
                ratio = (ti - 10) / 30
                temp[i] = T_underwater + ratio * (T_surface - T_underwater)
        
        else:
            # 后10s（40-50s）：维持目标温度
            if mode_name == 'Air to Surface':
                temp[i] = T_surface
            elif mode_name == 'Surface to Air':
                temp[i] = T_air
            elif mode_name == 'Air to Underwater':
                temp[i] = T_underwater
            elif mode_name == 'Underwater to Air':
                temp[i] = T_air
            elif mode_name == 'Surface to Underwater':
                temp[i] = T_underwater
            elif mode_name == 'Underwater to Surface':
                temp[i] = T_surface
    return temp

# 生成时间轴（0-50s，间隔0.1s，确保曲线平滑）
time = np.linspace(0, 50, 501)

# 创建图表（共享X轴，双Y轴）
fig, ax1 = plt.subplots(figsize=(14, 5))

# 创建共享X轴的第二个Y轴（温度）
ax2 = ax1.twinx()

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

# 绘制各模态的功率和温度曲线
for i, (mode_func, mode_name) in enumerate(modes):
    # 绘制功率曲线（实线）
    power = mode_func(time)
    ax1.plot(time, power, color=colors[i], label=f'{mode_name} (Power)', linewidth=2.5, alpha=0.8)
    
    # 绘制温度曲线（虚线）
    temp = get_temperature_curve(mode_name, time)
    ax2.plot(time, temp, color=colors[i], linestyle='--', label=f'{mode_name} (Temp)', linewidth=2, alpha=0.8)

# 添加阶段划分背景色（直观区分三个阶段）
ax1.axvspan(0, 10, alpha=0.1, color='gray', label='Pre-transition')
ax1.axvspan(10, 40, alpha=0.2, color='yellow', label='Transition')
ax1.axvspan(40, 50, alpha=0.1, color='green', label='Post-transition')

# 添加基础功率参考线（便于对比初始/目标功率）
ax1.axhline(y=P_air, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Air Power ({P_air}W)')
ax1.axhline(y=P_water_surface, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Surface Power ({P_water_surface}W)')
ax1.axhline(y=P_underwater, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Underwater Power ({P_underwater}W)')
# 添加2倍P_air参考线（验证水面/水下到空中的功率峰值）
ax1.axhline(y=2*P_air, color='red', linestyle='-.', linewidth=1.5, alpha=0.6, label=f'2×Air Power ({2*P_air}W)')

# 添加温度参考线
ax2.axhline(y=T_air, color='blue', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Air Temp ({T_air}℃)')
ax2.axhline(y=T_surface, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Surface Temp ({T_surface}℃)')
ax2.axhline(y=T_underwater, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Underwater Temp ({T_underwater}℃)')

# 设置功率轴（ax1）属性
ax1.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Power (W)', fontsize=14, fontweight='bold', color='black')
ax1.set_title('Power and Temperature Switching Process of Different Modes', fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 6000)  # 适配2倍P_air（5000W）的显示范围
ax1.tick_params(axis='y', labelsize=12, colors='black')
ax1.tick_params(axis='x', labelsize=12)

# 设置温度轴（ax2）属性
ax2.set_ylabel('Temperature (℃)', fontsize=14, fontweight='bold', color='darkred')
ax2.set_ylim(-5, 25)  # 温度范围：-5~25℃（覆盖所有场景）
ax2.tick_params(axis='y', labelsize=12, colors='darkred')

# 合并两个轴的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9, framealpha=0.9, shadow=True)

# 添加阶段标注（增强图表可读性）
ax1.text(5, ax1.get_ylim()[1] * 0.9, 'Pre-transition\n(0~10s)', ha='center', fontsize=11, 
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
ax1.text(25, ax1.get_ylim()[1] * 0.9, 'Transition\n(10~40s)', ha='center', fontsize=11,
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
ax1.text(45, ax1.get_ylim()[1] * 0.9, 'Post-transition\n(40~50s)', ha='center', fontsize=11,
        backgroundcolor='white', alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

# 美化图表边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# 调整布局（预留图例空间）
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 保存为SVG格式（高清无失真，适配论文排版）
plt.savefig('./Figures/Fig5-6 Power Switching Modes Design.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.savefig('./Figures/Fig5-6 Power Switching Modes Design.png', format='png', dpi=1200, bbox_inches='tight')

# 关闭图表释放资源
plt.close()

print("图表已保存至 ./Figures/Fig5-6 Power Switching Modes Design.svg")