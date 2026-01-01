import os
import sys
import time
import json
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入字体处理函数
from Scripts.Chapter3.MARL_Engine import Net, device

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, gaussian_kde

from Scripts.Chapter5.Env_Ultra import EnvUltra

# 延迟导入matplotlib，仅在需要时导入
def setup_matplotlib():
    """
    设置matplotlib环境
    """
    # 在需要时设置环境变量，确保动态链接器能找到正确的libstdc++版本
    conda_lib_path = '/home/nwpu/miniconda3/envs/Py310/lib'
    os.environ['LD_LIBRARY_PATH'] = conda_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # 确保使用正确的Python路径
    sys.path.insert(0, conda_lib_path)
    sys.path.insert(0, os.path.join(conda_lib_path, 'python3.10', 'site-packages'))
    
    # 导入matplotlib
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    
    # 导入字体处理函数并设置字体
    from Scripts.Chapter3.MARL_Engine import font_get
    font_get()
    
    return matplotlib, plt

# ----------------------------------------------------
# 1. 元强化学习环境类
# ----------------------------------------------------
class MetaRLEnvironment:
    """
    元强化学习环境，支持9种模态的生成和切换
    3种基础模态：空中飞行、水面航行、水下潜航
    6种切换模态：空中-水面、水面-空中、空中-水下、水下-空中、水面-水下、水下-水面
    """
    def __init__(self):
        self.base_modes = ['air', 'surface', 'underwater']
        self.switch_modes = [
            'air_to_surface', 'surface_to_air', 
            'air_to_underwater', 'underwater_to_air',
            'surface_to_underwater', 'underwater_to_surface'
        ]
        self.all_modes = self.base_modes + self.switch_modes
        
        # 基础模态的功率和温度参数（正态分布）
        self.mode_params = {
            'air': {
                'power': {'mean': 2500, 'std': 500},
                'temperature': {'mean': 0, 'std': 15}
            },
            'surface': {
                'power': {'mean': 1000, 'std': 300},
                'temperature': {'mean': 20, 'std': 5}
            },
            'underwater': {
                'power': {'mean': 3000, 'std': 400},
                'temperature': {'mean': 5, 'std': 3}
            }
        }
    
    def generate_mode_data(self, mode_type, duration=100):
        """
        生成指定模态的数据
        """
        if mode_type in self.base_modes:
            return self._generate_base_mode_data(mode_type, duration)
        else:
            # 切换模态需要起始和结束模态
            start_mode, end_mode = mode_type.split('_to_')
            return self._generate_switch_mode_data(start_mode, end_mode, duration)
    
    def _generate_base_mode_data(self, mode_type, duration):
        """
        生成基础模态数据
        """
        params = self.mode_params[mode_type]
        power = np.random.normal(params['power']['mean'], params['power']['std'], duration)
        power = np.maximum(power, 0)  # 功率不能为负
        temperature = np.random.normal(params['temperature']['mean'], params['temperature']['std'], duration)
        return {'power': power, 'temperature': temperature, 'mode': mode_type}
    
    def _generate_switch_mode_data(self, start_mode, end_mode, duration):
        """
        生成切换模态数据
        """
        start_params = self.mode_params[start_mode]
        end_params = self.mode_params[end_mode]
        
        # 线性过渡
        t = np.linspace(0, 1, duration)
        power = (1 - t) * np.random.normal(start_params['power']['mean'], start_params['power']['std'], duration) + \
                t * np.random.normal(end_params['power']['mean'], end_params['power']['std'], duration)
        temperature = (1 - t) * np.random.normal(start_params['temperature']['mean'], start_params['temperature']['std'], duration) + \
                      t * np.random.normal(end_params['temperature']['mean'], end_params['temperature']['std'], duration)
        
        return {'power': power, 'temperature': temperature, 'mode': f'{start_mode}_to_{end_mode}'}
    
    def generate_meta_task(self, mode_sequence):
        """
        生成元任务，由多个模态组成
        """
        task_data = {
            'power': np.array([]),
            'temperature': np.array([]),
            'mode': []
        }
        
        for mode in mode_sequence:
            mode_data = self.generate_mode_data(mode, duration=50)  # 每个模态持续50秒
            task_data['power'] = np.concatenate([task_data['power'], mode_data['power']])
            task_data['temperature'] = np.concatenate([task_data['temperature'], mode_data['temperature']])
            task_data['mode'].extend([mode] * 50)
        
        return task_data

# ----------------------------------------------------
# 2. 元强化学习策略类
# ----------------------------------------------------
class MetaRLPolicy(nn.Module):
    """
    元强化学习策略，基于RNN的策略网络，改进版
    """
    def __init__(self, input_dim=7, hidden_dim=256, num_layers=2):
        super(MetaRLPolicy, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # 添加多个特征层，提高模型表达能力
        self.fc_feature1 = nn.Linear(hidden_dim, 128)  # 第一特征层
        self.fc_feature2 = nn.Linear(128, 64)         # 第二特征层，与第四章一致
        self.fc_feature3 = nn.Linear(64, 32)          # 第三特征层
        
        # 分离的输出层，对应三个智能体
        self.fc_fc = nn.Linear(32, 32)  # 燃料电池动作空间
        self.fc_bat = nn.Linear(32, 40)  # 电池动作空间
        self.fc_sc = nn.Linear(32, 2)   # 超级电容动作空间
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        
        # 多层特征提取，使用ReLU激活函数和dropout
        feature1 = F.relu(self.fc_feature1(out))
        feature1 = F.dropout(feature1, p=0.2, training=self.training)
        
        feature2 = F.relu(self.fc_feature2(feature1))
        feature2 = F.dropout(feature2, p=0.2, training=self.training)
        
        feature3 = F.relu(self.fc_feature3(feature2))
        
        # 分别输出三个智能体的动作值
        fc_action = self.fc_fc(feature3)
        bat_action = self.fc_bat(feature3)
        sc_action = self.fc_sc(feature3)
        
        return fc_action, bat_action, sc_action, hidden

# ----------------------------------------------------
# 3. 快速适配机制
# ----------------------------------------------------
class FastAdapter:
    """
    快速适配机制，实现策略的在线更新
    """
    def __init__(self, meta_policy, kl_threshold=0.1):
        self.meta_policy = meta_policy
        self.kl_threshold = kl_threshold
    
    def compute_kl_divergence(self, current_dist, target_dist):
        """
        计算KL散度
        """
        # 使用核密度估计
        kde_current = gaussian_kde(current_dist)
        kde_target = gaussian_kde(target_dist)
        
        # 生成公共采样点
        min_val = min(current_dist.min(), target_dist.min())
        max_val = max(current_dist.max(), target_dist.max())
        x = np.linspace(min_val, max_val, 100)
        
        # 计算KL散度
        p = kde_current(x)
        q = kde_target(x)
        p = np.maximum(p, 1e-10)  # 避免除以零
        q = np.maximum(q, 1e-10)
        kl = np.sum(p * np.log(p / q)) * (x[1] - x[0])
        
        return kl
    
    def should_update(self, current_state, target_state):
        """
        判断是否需要更新策略
        """
        # 计算综合KL散度
        kl_power = self.compute_kl_divergence(current_state['power'], target_state['power'])
        kl_temp = self.compute_kl_divergence(current_state['temperature'], target_state['temperature'])
        kl_total = 0.5 * kl_power + 0.5 * kl_temp
        
        return kl_total >= self.kl_threshold
    
    def adapt(self, new_task_data, adaptation_steps=10):
        """
        快速适配到新任务
        """
        # 简化实现，实际需要基于新任务数据微调策略
        # 创建与原始模型相同参数的新模型
        # 获取原始模型的参数
        orig_input_dim = self.meta_policy.rnn.input_size
        orig_hidden_dim = self.meta_policy.rnn.hidden_size
        orig_num_layers = self.meta_policy.rnn.num_layers
        
        # 使用相同参数创建新模型
        adapted_policy = MetaRLPolicy(input_dim=orig_input_dim, hidden_dim=orig_hidden_dim, num_layers=orig_num_layers)
        adapted_policy.load_state_dict(self.meta_policy.state_dict())
        return adapted_policy

# ----------------------------------------------------
# 4. 基准策略：基于规则的策略
# ----------------------------------------------------
class RuleBasedPolicy:
    """
    基于规则的基准策略
    """
    def __init__(self):
        # 规则参数
        self.fc_threshold_high = 0.9  # 燃料电池高功率阈值
        self.fc_threshold_low = 0.4   # 燃料电池低功率阈值
        self.bat_soc_high = 0.8       # 电池SOC高阈值
        self.bat_soc_low = 0.2        # 电池SOC低阈值
    
    def choose_action(self, state):
        """
        基于规则选择动作
        state: [P_load, T_env, P_fc, P_bat, P_sc, soc_b, soc_sc]
        """
        P_load, T_env, P_fc, P_bat, P_sc, soc_b, soc_sc = state
        
        # 简化的规则策略
        if soc_b < self.bat_soc_low:
            # 电池SOC低，增加燃料电池输出
            fc_action = 15  # 增加燃料电池功率
        elif soc_b > self.bat_soc_high:
            # 电池SOC高，减少燃料电池输出
            fc_action = 0   # 减少燃料电池功率
        else:
            # 维持当前燃料电池输出
            fc_action = 8   # 中间值
        
        # 电池动作
        if P_load > P_fc:
            # 需要电池放电
            bat_action = 19  # 电池放电
        else:
            # 可以给电池充电
            bat_action = 0   # 电池充电
        
        # 超级电容动作
        sc_action = 1 if abs(P_load - P_fc - P_bat) > 500 else 0  # 功率差大时使用超级电容
        
        return [fc_action, bat_action, sc_action]

# ----------------------------------------------------
# 5. 基准策略：动态规划策略
# ----------------------------------------------------
class DPStrategy:
    """
    动态规划基准策略
    """
    def __init__(self, horizon=10):
        self.horizon = horizon  # 规划 horizon
    
    def choose_action(self, state):
        """
        基于动态规划选择动作
        简化实现，实际需要完整的DP算法
        """
        # 简化实现，返回固定动作
        return [8, 10, 1]

# ----------------------------------------------------
# 6. 结果保存类
# ----------------------------------------------------
class ResultSaver:
    """
    结果保存类，负责保存训练结果和测试结果
    """
    def __init__(self, base_path):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_model(self, model, model_name):
        """
        保存模型权重
        """
        model_path = os.path.join(self.base_path, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ 模型保存到: {model_path}")
        return model_path
    
    def save_results_json(self, results, filename="test_results.json"):
        """
        保存测试结果为JSON格式
        """
        json_path = os.path.join(self.base_path, filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            # 自定义JSON编码器，处理numpy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"✅ 测试结果保存到: {json_path}")
        return json_path
    
    def save_power_distribution_plot(self, power_data, scenario_name, filename="power_distribution.svg"):
        """
        生成并保存功率分配图为SVG格式，参照Chapter3的绘图逻辑
        """
        # 首先保存原始功率数据，方便后续处理
        data_path = os.path.join(self.base_path, f"power_data_{scenario_name}.npy")
        np.save(data_path, power_data)
        print(f"✅ 原始功率数据已保存到: {data_path}")
        
        # 延迟导入matplotlib，仅在需要时导入
        try:
            # 设置matplotlib环境
            matplotlib, plt = setup_matplotlib()
            
            # 从power_data中提取数据
            power_fc = power_data['power_fc']
            battery_power = power_data['power_bat']
            power_sc = power_data['power_sc']
            load_power = power_data['load_power']
            
            # 确保SOC数据存在
            soc_bat = power_data.get('soc_bat', [0.5] * len(power_fc))
            soc_sc_list = power_data.get('soc_sc', [0.5] * len(power_fc))
            temperature = power_data.get('temperature', [20.0] * len(power_fc))
            
            # 生成时间轴
            times = np.arange(len(power_fc))
            max_time = len(times)  # 实际测试时长
            
            # 绘图配置（完全参照Chapter3的设置）
            plt.rcParams.update({
                'font.family': ['Times New Roman'],  # 仅使用新罗马字体
                'axes.unicode_minus': False,
                'font.size': 12
            })
            
            # 统一的文章配色，与Chapter3完全一致
            article_color = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
            colors = article_color
            LINES_ALPHA = 1
            LABEL_FONT_SIZE = 18
            
            # ====================== 功率分配图绘制 ======================
            fig, ax1 = plt.subplots(figsize=(15, 5))
            fig.subplots_adjust(top=0.965, bottom=0.125, left=0.085, right=0.875)
            
            # 功率曲线，与Chapter3完全一致
            l1, = ax1.plot(times, load_power, label='Power Demand', color=colors[0], alpha=LINES_ALPHA)
            l2, = ax1.plot(times, power_fc, label='Power Fuel Cell', color=colors[1], alpha=LINES_ALPHA)
            l3, = ax1.plot(times, battery_power, label='Power Battery', color=colors[2], alpha=LINES_ALPHA)
            l6, = ax1.plot(times, power_sc, label='Power SuperCap', color='k', linestyle='--', alpha=LINES_ALPHA)

            # 坐标轴配置，与Chapter3完全一致
            ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
            ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
            ax1.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
            ax1.set_xlim(0, max_time)  # 使用实际测试时长
            ax1.set_ylim(-2500, 5500)  # 匹配功率峰值5000W

            # SOC曲线
            ax2 = ax1.twinx()
            l4, = ax2.plot(times, soc_bat, label='Battery SOC', color=colors[3], alpha=LINES_ALPHA)
            l7, = ax2.plot(times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':', alpha=LINES_ALPHA)
            ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE)
            ax2.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)
            ax2.set_ylim(0, 1.0)  # SOC范围0-1

            # 温度曲线
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 65))
            l5, = ax3.plot(times, temperature, label='Environment Temperature', color=colors[4], alpha=LINES_ALPHA)
            ax3.set_ylabel('Environment Temperature/°C', color=colors[4], fontsize=LABEL_FONT_SIZE)
            ax3.tick_params(axis='y', labelcolor=colors[4], labelsize=LABEL_FONT_SIZE)
            ax3.set_ylim(-25, 40)  # 温度范围

            # 根据场景类型获取实际工作模态信息
            env = EnvUltra(scenario_type=scenario_name)
            mode_annotations = env.mode_annotations
            
            # 模态到颜色的映射
            mode_colors = {
                'air': ('lightblue', 'Flight Phase'),
                'surface': ('lightgreen', 'Surface Sliding'),
                'underwater': ('salmon', 'Underwater Navigation'),
                'air_to_surface_switch': ('lightblue', 'Air to Surface'),
                'surface_to_air_switch': ('lightgreen', 'Surface to Air'),
                'air_to_underwater_switch': ('lightblue', 'Air to Underwater'),
                'underwater_to_surface_switch': ('salmon', 'Underwater to Surface'),
                'surface_to_underwater_switch': ('lightgreen', 'Surface to Underwater'),
                'underwater_to_air_switch': ('salmon', 'Underwater to Air')
            }
            
            # 绘制实际工作模态背景
            for mode in mode_annotations:
                start = mode['start']
                end = mode['end']
                mode_type = mode['type']
                
                # 确保模态类型在映射中存在
                if mode_type in mode_colors:
                    color, label = mode_colors[mode_type]
                    # 只添加一次标签，避免重复
                    if mode_annotations.index(mode) == 0:
                        ax1.axvspan(start, end, alpha=0.2, color=color, label=label)
                    else:
                        ax1.axvspan(start, end, alpha=0.2, color=color)

            # 图例配置，与Chapter3完全一致
            lines = [l1, l2, l3, l6, l4, l7, l5]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=LABEL_FONT_SIZE-2)
            ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)
            
            # 保存图像
            save_path_svg = os.path.join(self.base_path, filename)
            save_path_png = os.path.join(self.base_path, filename.replace('.svg', '.png'))
            
            plt.savefig(save_path_svg, bbox_inches='tight', dpi=1200)
            plt.savefig(save_path_png, dpi=1200, bbox_inches='tight')
            
            print(f"✅ 功率分配图保存到:")
            print(f"   SVG: {save_path_svg}")
            print(f"   PNG: {save_path_png}")
            
            # 关闭图像释放内存
            plt.close()
            
            return save_path_svg
        except Exception as e:
            print(f"⚠️  无法生成功率分配图: {e}")
            print(f"   原始功率数据已保存，但图表生成失败")
            return None

# ----------------------------------------------------
# 7. 工具函数
# ----------------------------------------------------
def get_project_root():
    """
    获取项目根目录
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def create_output_dir(subdir=""):
    """
    创建输出目录
    """
    project_root = get_project_root()
    timestamp = time.strftime("%m%d_%H%M%S")
    base_dir = os.path.join(project_root, "nets", "Chap5")
    if subdir:
        base_dir = os.path.join(base_dir, subdir)
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_model(policy, model_path):
    """
    加载模型权重
    """
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型: {model_path}")
        return True
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return False
