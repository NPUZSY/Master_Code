import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互模式，适配无GUI环境
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import sys
# 模拟字体设置（与Power_Profile.py一致）
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scripts.utils.global_utils import *
# 获取字体（优先宋体+Times New Roman，解决中文/负号显示）
font_get()

# 创建保存目录（若不存在）
os.makedirs('./Figures', exist_ok=True)

# 定义基础参数
TOTAL_DURATION = 1800  # 总时长1800s
SWITCH_DURATION = 50   # 切换模态时长50s
# 温度常量（℃）
T_AIR = 0
T_SURFACE = 20
T_UNDERWATER = 5
# 基础模态功率基准值（W）
P_AIR_BASE = 2500
P_SURFACE_BASE = 1000
P_UNDERWATER_BASE = 3000

# ===================== 工具函数 =====================
def generate_bayesian_power(time_points, base_power, scenario_type, mode_type):
    """
    基于贝叶斯模型生成带随机性的功率曲线
    :param time_points: 时间点数组
    :param base_power: 基础功率值
    :param scenario_type: 场景类型（cruise/recon/rescue）
    :param mode_type: 模态类型（air/surface/underwater）
    :return: 带随机波动的功率数组
    """
    # 不同场景的波动参数
    if scenario_type == 'cruise':  # 长航时巡航：小波动
        std = base_power * 0.05
    elif scenario_type == 'recon':  # 跨域侦察：水下大波动，其他小波动
        std = min(base_power * 0.15 if mode_type == 'underwater' else base_power * 0.05, 1000)
    elif scenario_type == 'rescue':  # 应急救援：整体大波动+峰值
        std = min(base_power * 0.2, 1000)
        peak_factor = 1.3  # 峰值系数
    else:
        std = base_power * 0.05

    # 生成随机波动
    np.random.seed(42)  # 固定随机种子保证可复现
    power_fluctuation = norm.rvs(loc=0, scale=std, size=len(time_points))
    power = base_power + power_fluctuation

    # 应急救援场景添加峰值
    if scenario_type == 'rescue':
        peak_indices = np.random.choice(len(time_points), size=int(len(time_points)*0.1), replace=False)
        power[peak_indices] = power[peak_indices] * peak_factor

    # 确保功率非负
    power = np.maximum(power, 0)
    return power

def generate_switch_power(time_points, start_power, end_power, switch_type):
    """
    生成切换模态的功率曲线（复用之前的切换逻辑）
    :param time_points: 切换阶段时间点（0~50s）
    :param start_power: 起始模态功率
    :param end_power: 目标模态功率
    :param switch_type: 切换类型（air_to_surface/surface_to_air/air_to_underwater/underwater_to_surface等）
    :return: 切换阶段功率数组
    """
    power = np.zeros_like(time_points)
    for i, ti in enumerate(time_points):
        if ti <= 10:
            power[i] = start_power
        elif ti <= 20:
            if switch_type in ['air_to_surface', 'surface_to_air', 'air_to_underwater', 'underwater_to_air']:
                ratio = (ti - 10) / 10
                if switch_type == 'air_to_surface':
                    power[i] = start_power + ratio * (1.1 * start_power - start_power)
                elif switch_type == 'surface_to_air':
                    power[i] = start_power + ratio * (start_power - start_power)
                elif switch_type == 'air_to_underwater':
                    power[i] = start_power
                elif switch_type == 'underwater_to_air':
                    power[i] = start_power + ratio * (1.1 * start_power - start_power)
        elif ti <= 35:
            ratio = (ti - 20) / 15
            if switch_type == 'air_to_surface':
                power[i] = 1.1 * start_power + ratio * (end_power - 1.1 * start_power)
            elif switch_type == 'surface_to_air':
                power[i] = start_power + ratio * (end_power - start_power)
            elif switch_type == 'air_to_underwater':
                power[i] = start_power
            elif switch_type == 'underwater_to_air':
                power[i] = 1.1 * start_power + ratio * (end_power - 1.1 * start_power)
            elif switch_type == 'underwater_to_surface':
                power[i] = start_power + ratio * (end_power - start_power)
        elif ti <= 40:
            ratio = (ti - 35) / 5
            if switch_type == 'air_to_surface':
                power[i] = end_power
            elif switch_type == 'surface_to_air':
                power[i] = end_power
            elif switch_type == 'air_to_underwater':
                ratio_full = (ti - 25) / 10 if ti > 25 else 0
                power[i] = end_power
            elif switch_type == 'underwater_to_surface':
                power[i] = end_power
        else:
            power[i] = end_power
    return power

def generate_temperature_curve(time_points, mode_sequence, scenario_type):
    """
    生成温度变化曲线
    :param time_points: 时间点数组
    :param mode_sequence: 模态序列（包含基础模态和切换模态）
    :param scenario_type: 场景类型
    :return: 温度数组
    """
    temp = np.zeros_like(time_points, dtype=np.float64)
    time_idx = 0
    
    # 温度变化速率系数（应急救援更快）
    rate_factor = 1.5 if scenario_type == 'rescue' else 1.0

    for segment in mode_sequence:
        seg_type, seg_start, seg_end = segment['type'], segment['start'], segment['end']
        seg_len = seg_end - seg_start
        seg_time = time_points[time_idx:time_idx+seg_len]
        
        if seg_type == 'air':
            # 空中温度：0℃，带小幅波动
            seg_temp = np.full(seg_len, T_AIR, dtype=np.float64)
        elif seg_type == 'surface':
            # 水面温度：20℃，带小幅波动
            seg_temp = np.full(seg_len, T_SURFACE, dtype=np.float64)
        elif seg_type == 'underwater':
            # 水下温度：5℃，带小幅波动
            seg_temp = np.full(seg_len, T_UNDERWATER, dtype=np.float64)
        elif 'switch' in seg_type:
            # 切换模态温度变化
            seg_temp = np.zeros(seg_len, dtype=np.float64)
            switch_time = np.linspace(0, SWITCH_DURATION, seg_len)
            
            for i, ti in enumerate(switch_time):
                if ti <= 10:
                    # 前10s维持初始温度
                    if seg_type == 'air_to_surface_switch':
                        seg_temp[i] = T_AIR
                    elif seg_type == 'surface_to_air_switch':
                        seg_temp[i] = T_SURFACE
                    elif seg_type == 'air_to_underwater_switch':
                        seg_temp[i] = T_AIR
                    elif seg_type == 'underwater_to_surface_switch':
                        seg_temp[i] = T_UNDERWATER
                    elif seg_type == 'surface_to_underwater_switch':
                        seg_temp[i] = T_SURFACE
                    elif seg_type == 'underwater_to_air_switch':
                        seg_temp[i] = T_UNDERWATER
                elif 10 < ti <= 40:
                    # 过渡阶段线性变化（应急救援速率更快）
                    ratio = ((ti - 10) / 30) * rate_factor
                    ratio = np.clip(ratio, 0, 1)  # 防止超出范围
                    
                    if seg_type == 'air_to_surface_switch':
                        seg_temp[i] = T_AIR + ratio * (T_SURFACE - T_AIR)
                    elif seg_type == 'surface_to_air_switch':
                        seg_temp[i] = T_SURFACE - ratio * (T_SURFACE - T_AIR)
                    elif seg_type == 'air_to_underwater_switch':
                        # 先升后降（25s达20℃）
                        if ti <= 25:
                            ratio_sub = ((ti - 10) / 15) * rate_factor
                            seg_temp[i] = T_AIR + ratio_sub * (T_SURFACE - T_AIR)
                        else:
                            ratio_sub = ((ti - 25) / 15) * rate_factor
                            seg_temp[i] = T_SURFACE - ratio_sub * (T_SURFACE - T_UNDERWATER)
                    elif seg_type == 'underwater_to_surface_switch':
                        seg_temp[i] = T_UNDERWATER + ratio * (T_SURFACE - T_UNDERWATER)
                    elif seg_type == 'surface_to_underwater_switch':
                        seg_temp[i] = T_SURFACE - ratio * (T_SURFACE - T_UNDERWATER)
                    elif seg_type == 'underwater_to_air_switch':
                        # 先升后降（25s达20℃）
                        if ti <= 25:
                            ratio_sub = ((ti - 10) / 15) * rate_factor
                            seg_temp[i] = T_UNDERWATER + ratio_sub * (T_SURFACE - T_UNDERWATER)
                        else:
                            ratio_sub = ((ti - 25) / 15) * rate_factor
                            seg_temp[i] = T_SURFACE - ratio_sub * (T_SURFACE - T_AIR)
                else:
                    # 后10s维持目标温度
                    if seg_type == 'air_to_surface_switch':
                        seg_temp[i] = T_SURFACE
                    elif seg_type == 'surface_to_air_switch':
                        seg_temp[i] = T_AIR
                    elif seg_type == 'air_to_underwater_switch':
                        seg_temp[i] = T_UNDERWATER
                    elif seg_type == 'underwater_to_surface_switch':
                        seg_temp[i] = T_SURFACE
                    elif seg_type == 'surface_to_underwater_switch':
                        seg_temp[i] = T_UNDERWATER
                    elif seg_type == 'underwater_to_air_switch':
                        seg_temp[i] = T_AIR
        else:
            seg_temp = np.zeros(seg_len, dtype=np.float64)
        
        # 添加温度波动
        np.random.seed(42)
        temp_fluctuation = norm.rvs(loc=0, scale=0.5*rate_factor, size=seg_len)
        seg_temp += temp_fluctuation
        seg_temp = np.clip(seg_temp, -5, 25)  # 温度范围限制
        
        temp[time_idx:time_idx+seg_len] = seg_temp
        time_idx += seg_len
    
    return temp

def build_scenario_profile(scenario_type):
    """
    构建不同场景的功率和温度剖面
    :param scenario_type: cruise/recon/rescue
    :return: time_points, power_profile, temp_profile, mode_annotations
    """
    # 生成时间轴（1800s，1s间隔）
    time_points = np.arange(0, TOTAL_DURATION, 1)
    power_profile = np.zeros_like(time_points, dtype=float)
    mode_annotations = []  # 存储模态标注信息
    
    if scenario_type == 'cruise':
        # 长航时巡航：空中(0-600)→切换(600-650)→水面(650-1150)→切换(1150-1200)→空中(1200-1800)
        # 阶段1：空中飞行 0-600s
        air1_time = time_points[0:600]
        air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'cruise', 'air')
        power_profile[0:600] = air1_power
        mode_annotations.append({'type': 'air', 'start': 0, 'end': 600, 'label': 'Air Flight'})
        
        # 阶段2：空中→水面切换 600-650s
        switch1_time = time_points[600:650]
        switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_SURFACE_BASE, 'air_to_surface')
        power_profile[600:650] = switch1_power
        mode_annotations.append({'type': 'air_to_surface_switch', 'start': 600, 'end': 650, 'label': 'Air→Surface Switch'})
        
        # 阶段3：水面航行 650-1150s (500s)
        surface_time = time_points[650:1150]
        surface_power = generate_bayesian_power(surface_time, P_SURFACE_BASE, 'cruise', 'surface')
        power_profile[650:1150] = surface_power
        mode_annotations.append({'type': 'surface', 'start': 650, 'end': 1150, 'label': 'Surface Navigation'})
        
        # 阶段4：水面→空中切换 1150-1200s
        switch2_time = time_points[1150:1200]
        switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
        power_profile[1150:1200] = switch2_power
        mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1150, 'end': 1200, 'label': 'Surface→Air Switch'})
        
        # 阶段5：空中飞行 1200-1800s
        air2_time = time_points[1200:1800]
        air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'cruise', 'air')
        power_profile[1200:1800] = air2_power
        mode_annotations.append({'type': 'air', 'start': 1200, 'end': 1800, 'label': 'Air Flight'})
        
    elif scenario_type == 'recon':
        # 跨域侦察：空中(0-200)→切换(200-250)→水下(250-1300)→切换(1300-1350)→水面(1350-1550)→切换(1550-1600)→空中(1600-1800)
        # 阶段1：空中飞行 0-200s
        air1_time = time_points[0:200]
        air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'recon', 'air')
        power_profile[0:200] = air1_power
        mode_annotations.append({'type': 'air', 'start': 0, 'end': 200, 'label': 'Air Flight'})
        
        # 阶段2：空中→水下切换 200-250s
        switch1_time = time_points[200:250]
        switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_UNDERWATER_BASE, 'air_to_underwater')
        power_profile[200:250] = switch1_power
        mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 200, 'end': 250, 'label': 'Air→Underwater Switch'})
        
        # 阶段3：水下潜航 250-1300s (1050s)
        underwater_time = time_points[250:1300]
        underwater_power = generate_bayesian_power(underwater_time, P_UNDERWATER_BASE, 'recon', 'underwater')
        power_profile[250:1300] = underwater_power
        mode_annotations.append({'type': 'underwater', 'start': 250, 'end': 1300, 'label': 'Underwater Navigation'})
        
        # 阶段4：水下→水面切换 1300-1350s
        switch2_time = time_points[1300:1350]
        switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_UNDERWATER_BASE, P_SURFACE_BASE, 'underwater_to_surface')
        power_profile[1300:1350] = switch2_power
        mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1300, 'end': 1350, 'label': 'Underwater→Surface Switch'})
        
        # 阶段5：水面航行 1350-1550s (200s)
        surface_time = time_points[1350:1550]
        surface_power = generate_bayesian_power(surface_time, P_SURFACE_BASE, 'recon', 'surface')
        power_profile[1350:1550] = surface_power
        mode_annotations.append({'type': 'surface', 'start': 1350, 'end': 1550, 'label': 'Surface Navigation'})
        
        # 阶段6：水面→空中切换 1550-1600s
        switch3_time = time_points[1550:1600]
        switch3_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
        power_profile[1550:1600] = switch3_power
        mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1550, 'end': 1600, 'label': 'Surface→Air Switch'})
        
        # 阶段7：空中飞行 1600-1800s
        air2_time = time_points[1600:1800]
        air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'recon', 'air')
        power_profile[1600:1800] = air2_power
        mode_annotations.append({'type': 'air', 'start': 1600, 'end': 1800, 'label': 'Air Flight'})
        
    elif scenario_type == 'rescue':
        # 应急救援：水面(0-320)→切换(320-370)→空中(370-690)→切换(690-740)→水下(740-1060)→切换(1060-1110)→水面(1110-1430)→切换(1430-1480)→空中(1480-1800)
        # 阶段1：水面航行 0-320s
        surface1_time = time_points[0:320]
        surface1_power = generate_bayesian_power(surface1_time, P_SURFACE_BASE, 'rescue', 'surface')
        power_profile[0:320] = surface1_power
        mode_annotations.append({'type': 'surface', 'start': 0, 'end': 320, 'label': 'Surface Navigation'})
        
        # 阶段2：水面→空中切换 320-370s
        switch1_time = time_points[320:370]
        switch1_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
        power_profile[320:370] = switch1_power
        mode_annotations.append({'type': 'surface_to_air_switch', 'start': 320, 'end': 370, 'label': 'Surface→Air Switch'})
        
        # 阶段3：空中飞行 370-690s (320s)
        air1_time = time_points[370:690]
        air1_power = generate_bayesian_power(air1_time, P_AIR_BASE, 'rescue', 'air')
        power_profile[370:690] = air1_power
        mode_annotations.append({'type': 'air', 'start': 370, 'end': 690, 'label': 'Air Flight'})
        
        # 阶段4：空中→水下切换 690-740s
        switch2_time = time_points[690:740]
        switch2_power = generate_switch_power(np.linspace(0, 50, 50), P_AIR_BASE, P_UNDERWATER_BASE, 'air_to_underwater')
        power_profile[690:740] = switch2_power
        mode_annotations.append({'type': 'air_to_underwater_switch', 'start': 690, 'end': 740, 'label': 'Air→Underwater Switch'})
        
        # 阶段5：水下潜航 740-1060s (320s)
        underwater_time = time_points[740:1060]
        underwater_power = generate_bayesian_power(underwater_time, P_UNDERWATER_BASE, 'rescue', 'underwater')
        power_profile[740:1060] = underwater_power
        mode_annotations.append({'type': 'underwater', 'start': 740, 'end': 1060, 'label': 'Underwater Navigation'})
        
        # 阶段6：水下→水面切换 1060-1110s
        switch3_time = time_points[1060:1110]
        switch3_power = generate_switch_power(np.linspace(0, 50, 50), P_UNDERWATER_BASE, P_SURFACE_BASE, 'underwater_to_surface')
        power_profile[1060:1110] = switch3_power
        mode_annotations.append({'type': 'underwater_to_surface_switch', 'start': 1060, 'end': 1110, 'label': 'Underwater→Surface Switch'})
        
        # 阶段7：水面航行 1110-1430s (320s)
        surface2_time = time_points[1110:1430]
        surface2_power = generate_bayesian_power(surface2_time, P_SURFACE_BASE, 'rescue', 'surface')
        power_profile[1110:1430] = surface2_power
        mode_annotations.append({'type': 'surface', 'start': 1110, 'end': 1430, 'label': 'Surface Navigation'})
        
        # 阶段8：水面→空中切换 1430-1480s
        switch4_time = time_points[1430:1480]
        switch4_power = generate_switch_power(np.linspace(0, 50, 50), P_SURFACE_BASE, P_AIR_BASE, 'surface_to_air')
        power_profile[1430:1480] = switch4_power
        mode_annotations.append({'type': 'surface_to_air_switch', 'start': 1430, 'end': 1480, 'label': 'Surface→Air Switch'})
        
        # 阶段9：空中飞行 1480-1800s
        air2_time = time_points[1480:1800]
        air2_power = generate_bayesian_power(air2_time, P_AIR_BASE, 'rescue', 'air')
        power_profile[1480:1800] = air2_power
        mode_annotations.append({'type': 'air', 'start': 1480, 'end': 1800, 'label': 'Air Flight'})
    
    # 生成温度曲线
    temp_profile = generate_temperature_curve(time_points, mode_annotations, scenario_type)
    
    return time_points, power_profile, temp_profile, mode_annotations

# ===================== 绘图主函数 =====================
def plot_scenario_profiles():
    """绘制三种场景的功率和温度剖面"""
    # 创建子图（3行1列，共享X轴）
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Power and Temperature Profiles of Typical Scenarios', fontsize=18, fontweight='bold', y=0.98)
    
    # 定义场景配置
    scenarios = [
        ('cruise', 'Long-Endurance Cruise', '#1f77b4'),
        ('recon', 'Cross-Domain Reconnaissance', '#ff7f0e'),
        ('rescue', 'Emergency Rescue', '#2ca02c')
    ]
    
    # 绘制每个场景
    for idx, (scenario_type, scenario_label, color) in enumerate(scenarios):
        ax1 = axes[idx]
        ax2 = ax1.twinx()  # 共享X轴的温度轴
        
        # 构建场景剖面
        time, power, temp, modes = build_scenario_profile(scenario_type)
        
        # 绘制功率曲线
        ax1.plot(time, power, color=color, linewidth=1.2, label='Power Demand')
        ax1.fill_between(time, 0, power, color=color, alpha=0.1)
        
        # 绘制温度曲线（虚线）
        ax2.plot(time, temp, color='darkred', linestyle='--', linewidth=1.2, label='Temperature')
        
        # 标注模态阶段
        for mode in modes:
            # 绘制模态背景色
            if 'air' in mode['type'] and 'switch' not in mode['type']:
                ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightblue')
            elif 'surface' in mode['type'] and 'switch' not in mode['type']:
                ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightyellow')
            elif 'underwater' in mode['type'] and 'switch' not in mode['type']:
                ax1.axvspan(mode['start'], mode['end'], alpha=0.1, color='lightgreen')
            elif 'switch' in mode['type']:
                ax1.axvspan(mode['start'], mode['end'], alpha=0.2, color='orange')
            
            # 添加模态标签（仅标注主要模态）
            if 'switch' not in mode['type']:
                mid_time = (mode['start'] + mode['end']) / 2
                ax1.text(mid_time, ax1.get_ylim()[1]*0.7, mode['label'], 
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 设置轴属性
        ax1.set_title(scenario_label, fontsize=14, fontweight='bold', pad=10)
        ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, max(power)*1.1)
        ax1.tick_params(axis='y', labelsize=10)
        
        ax2.set_ylabel('Temperature (℃)', fontsize=12, fontweight='bold', color='darkred')
        ax2.set_ylim(-5, 25)
        ax2.tick_params(axis='y', labelsize=10, colors='darkred')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10, framealpha=0.9)
        
        # 美化边框
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    
    # 设置X轴
    axes[-1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    axes[-1].set_xlim(0, TOTAL_DURATION)
    axes[-1].set_xticks(np.arange(0, TOTAL_DURATION+1, 200))
    axes[-1].tick_params(axis='x', labelsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存SVG文件
    plt.savefig('./Figures/Fig5-7 Profiles.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("图表已保存至 ./Figures/Fig5-7 Profiles.svg")

# ===================== 执行绘图 =====================
if __name__ == '__main__':
    plot_scenario_profiles()