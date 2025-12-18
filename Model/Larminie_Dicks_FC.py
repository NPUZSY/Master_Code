import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from opem.Static.Larminie_Dicks import Static_Analysis
from opem.Dynamic.Chakraborty import Dynamic_Analysis
def font_get():
    """
    加载Times New Roman和宋体(SimSun)字体，确保Matplotlib正常显示中英文
    - Times New Roman路径: /usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf
    - 宋体路径: /usr/share/fonts/truetype/simsun.ttc
    """
    import os
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 定义字体路径
    tnr_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    simsun_font_path = '/usr/share/fonts/truetype/simsun.ttc'

    # 加载Times New Roman字体
    if os.path.exists(tnr_font_path):
        fm.fontManager.addfont(tnr_font_path)
    else:
        print(f"警告：Times New Roman字体文件不存在: {tnr_font_path}")

    # 加载宋体(SimSun)字体
    if os.path.exists(simsun_font_path):
        fm.fontManager.addfont(simsun_font_path)
    else:
        print(f"警告：宋体字体文件不存在: {simsun_font_path}")

    # 方案2（可选）：优先Times New Roman（适合英文为主的场景）
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],
        'font.sans-serif': ['Times New Roman', 'SimSun'],
        'axes.unicode_minus': False,
        'font.size': 12
    })

    # 验证字体加载结果
    try:
        # 检查Times New Roman
        tnr_fp = fm.FontProperties(family='Times New Roman')
        tnr_loaded = 'Times_New_Roman' in fm.findfont(tnr_fp)
        # 检查宋体
        simsun_fp = fm.FontProperties(family='SimSun')
        simsun_loaded = 'simsun' in fm.findfont(simsun_fp).lower()
        
    except Exception as e:
        print(f"字体验证失败: {e}")
font_get()

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

def larminie_dicks_efficiency(power, T=298):
    """
    根据Larminie-Dicks模型，通过输入温度和功率计算燃料电池效率。

    参数:
    power (float): 燃料电池的输出功率，单位为瓦特
    T (float): 电池操作温度，单位为K

    返回:
    float: 燃料电池的效率
    """
    # 根据功率和假设电压计算电流（此处先保留，但后续建议按模型正确计算电压来获取电流）
    N_number = 500
    voltage_assumed = 1.3
    current = power / voltage_assumed / N_number

    # 根据 Larminie-Dicks模型的参数要求构造测试向量
    Test_Vector = {
        "A": 0.06,
        "E0": 1.178,
        "T": T,
        "RM": 0.0018,
        "i_0": 0.00654,
        "i_L": 100.0,
        "i_n": 0.23,
        "N": N_number,
        "i-start": current,
        "i-stop": current + 0.01,
        "i-step": 0.01,
        "Name": "Larminiee_Test"
    }
    # print("Constructed Test_Vector:", Test_Vector)  # 打印构建的测试向量信息

    # 调用 Larminie-Dicks模型的静态分析函数获取结果
    data = Static_Analysis(InputMethod=Test_Vector, TestMode=True, PrintMode=False, ReportMode=False)
    # print("Data obtained from Static_Analysis:", data)  # 打印从静态分析函数得到的数据

    # 返回计算得到的效率值，如果计算过程中出现问题（比如数据格式不对等），可能需要根据实际情况调整错误处理逻辑
    return data["EFF"][0]


# 将摄氏度转换为开尔文温度
def celsius_to_kelvin(celsius):
    return celsius + 273.15


# 将开尔文温度转换为摄氏度
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15


def plot3D():
    # 温度范围（从 -90 摄氏度到 25 摄氏度，转换为开尔文）
    T_range_kelvin = np.array([celsius_to_kelvin(t) for t in np.arange(-25, 26, 1)])
    # 功率范围（从 0W 到 7000W）
    power_range = np.arange(0, 8001, 10)

    # 生成网格数据
    T_mesh, power_mesh = np.meshgrid(T_range_kelvin, power_range)
    efficiency_mesh = np.zeros_like(T_mesh)

    # 计算总网格点数量，用于进度计算
    total_points = T_mesh.shape[0] * T_mesh.shape[1]
    completed_points = 0

    # 标记是否从文件读取数据，初始化为False，表示先进行计算
    read_from_file = False

    if read_from_file:
        # 从本地文件读取数据
        data = np.load('../Data/fuel_cell_data.npz')
        T_mesh = data['T_mesh']
        power_mesh = data['power_mesh']
        efficiency_mesh = data['efficiency_mesh']
    else:
        # 使用tqdm创建进度条
        with tqdm(total=total_points) as pbar:
            # 计算每个网格点对应的效率值
            for i in range(T_mesh.shape[0]):
                for j in range(T_mesh.shape[1]):
                    efficiency_mesh[i, j] = larminie_dicks_efficiency(power_mesh[i, j], T_mesh[i, j])
                    completed_points += 1
                    # 更新进度条进度，显示百分比
                    pbar.update(1)
                    pbar.set_description(f"Progress: {completed_points / total_points * 100:.2f}%")

        # 将计算后的数据保存到本地文件
        np.savez('./Data/fuel_cell_data.npz', T_mesh=T_mesh, power_mesh=power_mesh, efficiency_mesh=efficiency_mesh)

    # 创建三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=50)
    # 将T_mesh中的开尔文温度转换为摄氏度用于绘图显示
    T_mesh_celsius = kelvin_to_celsius(T_mesh)

    # 绘制三维曲面图，使用'coolwarm'配色方案（你提到的colorwarm可能有误，这里推测是类似的'coolwarm'）
    surf = ax.plot_surface(T_mesh_celsius, power_mesh, efficiency_mesh, cmap='coolwarm')

    # 设置坐标轴标签，将温度轴标签改为摄氏度表示
    ax.set_xlabel('Temperature / °C')
    ax.set_ylabel('Power / W')
    ax.set_zlabel('Efficiency')

    # # 添加颜色条
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    fig.savefig('./Figures/Fig2-3 FC_Efficiency_EN.svg')

    # 显示图形
    plt.show()


def plot():
    # 功率范围（从0W到700W）
    current_range = np.arange(0, 5000, 0.1)
    efficiencies = []
    power_list = []
    voltage_list = []
    T = 298.15
    for power in current_range:
        # 根据 Larminie-Dicks模型计算每个功率下的效率
        efficiency = larminie_dicks_efficiency(power, T)
        efficiencies.append(efficiency)
        power_list.append(power)
        # voltage_list.append(voltage)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制效率随功率变化的曲线
    ax.plot(power_list, efficiencies, label='Fuel Cell Efficiency')

    # 设置坐标轴标签
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Efficiency')
    #
    # # 创建图形对象和主坐标轴对象
    # fig, ax1 = plt.subplots(figsize=(8, 6))
    #
    # # 绘制功率随电流变化的曲线，使用主坐标轴（左侧Y轴）
    # color = 'tab:blue'
    # ax1.set_xlabel('Current Density (A/cm2)')
    # ax1.set_ylabel('Power (W)', color=color)
    # ax1.plot(current_range, power_list, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # # 创建与主坐标轴共享X轴的次坐标轴对象（用于绘制效率曲线，右侧第一个Y轴）
    # ax2 = ax1.twinx()
    # color = 'tab:red'
    # ax2.set_ylabel('Efficiency', color=color)
    # ax2.plot(current_range, efficiencies, label='efficiency', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim(0, 1)  # 可根据效率合理范围设置y轴范围，这里示例设置为0到1

    # 设置图形标题
    ax.set_title(f'Fuel Cell Efficiency at Temperature {kelvin_to_celsius(T)} °C')
    # ax1.set_title(f'Fuel Cell Characteristics at Temperature {kelvin_to_celsius(T)} °C')

    # 添加网格线
    plt.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5)
    fig.savefig('../Figures/FC_Efficiency_2D.svg')
    plt.show()


if __name__ == '__main__':
    plot3D()
    # plot()