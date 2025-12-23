import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import json
import argparse
import sys
from json import JSONEncoder
import torch.nn as nn
import torch.nn.functional as F
import shap
import pandas as pd
import re  # 正则表达式处理数值


# 示例指令
# python Scripts/Chapter4/shap_analysis.py --net-date 1222 --train-id 11 --n-samples 500


# ====================== 1. 环境与路径配置（复用原有逻辑） ======================
def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# 导入原有引擎组件
from Scripts.Chapter3.MARL_Engine import Net, IndependentDQN, device
from Scripts.Env import Envs  # 导入你提供的真实环境
from Scripts.utils.global_utils import font_get

# 获取字体设置
font_get()

# ====================== 2. JointNet 相关类定义（完全复用） ======================
class MultiTaskRNN(nn.Module):
    """适配 7 维输入的多任务 RNN 结构"""
    def __init__(self, input_dim=7, hidden_dim_rnn=256, num_layers_rnn=2, hidden_dim_fc=64):
        super(MultiTaskRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim_rnn, num_layers=num_layers_rnn, batch_first=True)
        self.fc_rnn_to_64 = nn.Linear(hidden_dim_rnn, hidden_dim_fc)
        self.reg_head = nn.Linear(hidden_dim_fc, 1)
        self.cls_head = nn.Linear(hidden_dim_fc, 4)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn[:, -1, :]
        feature_64 = F.relu(self.fc_rnn_to_64(out_rnn))
        return self.reg_head(feature_64), self.cls_head(feature_64), feature_64

class JointNet(nn.Module):
    """拼接 RNN 特征(64) + 回归值(1) = 65维输入 MARL Head"""
    def __init__(self, rnn_part, marl_head):
        super(JointNet, self).__init__()
        self.rnn_part = rnn_part
        self.marl_part = marl_head

    def forward(self, x):
        reg_out, _, feature_64 = self.rnn_part(x)
        joint_input = torch.cat([feature_64, reg_out], dim=1)
        return self.marl_part(joint_input)

class JointDQN(IndependentDQN):
    """支持 7 维输入并自动执行内部拼接的智能体"""
    def __init__(self, agent_name, rnn_model, n_actions):
        super(JointDQN, self).__init__(agent_name, 65, n_actions)
        self.n_actions = n_actions
        self.eval_net = JointNet(rnn_model, self.eval_net).to(device)
        self.target_net = JointNet(rnn_model, self.target_net).to(device)

    def predict_q_values(self, x):
        """适配SHAP的预测函数：输入状态矩阵，输出所有动作的Q值"""
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            q_values = self.eval_net(x_tensor)
        return q_values.cpu().numpy()

    def predict_max_q(self, x):
        """适配SHAP的预测函数：输入状态矩阵，输出max Q值（决策对应的Q值）"""
        q_values = self.predict_q_values(x)
        return np.max(q_values, axis=1)

# ====================== 3. 工具类与参数解析 ======================
class NumpyEncoder(JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='JointNet智能体SHAP分析脚本（归一化横坐标的整合Dependence Plot）')
    
    # 核心参数
    parser.add_argument('--net-date', type=str, required=True,
                        help='模型所在的日期文件夹（必填，如：1213）')
    parser.add_argument('--train-id', type=str, required=True,
                        help='模型对应的训练ID（必填，如：11）')
    parser.add_argument('--rnn-path', type=str, 
                        default=os.path.join(project_root, "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth"),
                        help='预训练RNN模型路径')
    
    # 可选配置
    parser.add_argument('--model-prefix', type=str, default="Joint_Model", help='模型前缀')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--n-samples', type=int, default=500, 
                        help='SHAP分析的采样数量（默认500，越多越准确但耗时更长）')
    parser.add_argument('--show-plot', action='store_true', help='是否显示SHAP图（默认仅保存）')
    
    return parser.parse_args()

# ====================== 4. 数据归一化工具函数 ======================
def min_max_normalize(data, min_val=None, max_val=None):
    """
    Min-Max归一化：将数据缩放到[0,1]区间
    参数:
        data: 待归一化的数组
        min_val: 手动指定最小值（None则自动计算）
        max_val: 手动指定最大值（None则自动计算）
    返回:
        normalized_data: 归一化后的数据
        min_val: 使用的最小值
        max_val: 使用的最大值
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    # 避免除以0
    if max_val - min_val < 1e-8:
        return np.zeros_like(data), min_val, max_val
    
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

# ====================== 5. SHAP分析核心函数（归一化横坐标） ======================
def generate_state_samples(env, n_samples=500):
    """
    生成覆盖所有状态维度的采样数据集（严格匹配环境的observation_space）
    状态维度定义（来自Envs类的observation_space）：
    [0] P_load: 负载功率 (0 ~ 80000 W)
    [1] Temperature: 环境温度 (-100 ~ 200 °C)
    [2] P_fc: 燃料电池功率 (0 ~ 5000 W)
    [3] P_bat: 电池功率 (-5000 ~ 5000 W)
    [4] P_sc: 超级电容功率 (-2000 ~ 2000 W)
    [5] SOC_bat: 电池SOC (0 ~ 1)
    [6] SOC_sc: 超级电容SOC (0 ~ 1)
    """
    samples = []
    
    # 获取环境的观测空间上下限
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    
    # 覆盖真实工况的采样（基于环境实际参数）
    for _ in range(n_samples):
        # 1. 负载功率：从真实loads数据中采样，回退到均匀分布
        if len(env.loads) > 0:
            p_load = float(np.random.choice(env.loads))
        else:
            p_load = np.random.uniform(obs_low[0], obs_high[0])
        
        # 2. 环境温度：从真实temperature数据中采样，回退到均匀分布
        if len(env.temperature) > 0:
            temp = float(np.random.choice(env.temperature))
        else:
            temp = np.random.uniform(obs_low[1], obs_high[1])
        
        # 3. 燃料电池功率：0 ~ P_FC_MAX(5000W)
        p_fc = np.random.uniform(obs_low[2], env.P_FC_MAX)
        
        # 4. 电池功率：-P_BAT_MAX(5000W) ~ P_BAT_MAX(5000W)
        p_bat = np.random.uniform(-env.P_BAT_MAX, env.P_BAT_MAX)
        
        # 5. 超级电容功率：-P_SC_MAX(2000W) ~ P_SC_MAX(2000W)
        p_sc = np.random.uniform(-env.P_SC_MAX, env.P_SC_MAX)
        
        # 6. 电池SOC：0.2 ~ 0.8（环境中惩罚区间外的合理范围）
        soc_bat = np.random.uniform(0.2, 0.8)
        
        # 7. 超级电容SOC：0 ~ 1
        soc_sc = np.random.uniform(obs_low[6], obs_high[6])
        
        # 构造状态向量
        state = np.array([
            p_load, temp, p_fc, p_bat, p_sc, soc_bat, soc_sc
        ], dtype=np.float32)
        
        # 确保状态在观测空间范围内
        state = np.clip(state, obs_low, obs_high)
        samples.append(state)
    
    return np.array(samples)

def plot_combined_dependence_normalized(shap_values, state_samples, feature_names, top_k=3, 
                                        agent_name="Agent", save_dir="./", show_plot=False):
    """
    绘制归一化横坐标的整合Dependence Plot：
    - 每个特征的横坐标先Min-Max归一化到[0,1]
    - 图例中标注原始取值范围，保证物理意义
    """
    # 定义颜色和标记（区分不同特征）
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # 红、蓝、绿
    markers = ['o', 's', '^']  # 圆形、方形、三角形
    
    # 创建画布
    plt.figure(figsize=(12, 8))
    
    # 遍历TOP K特征
    for i in range(top_k):
        # 提取该特征的原始取值
        feature_vals_original = state_samples[:, i]
        # 对特征取值进行Min-Max归一化
        feature_vals_norm, min_val, max_val = min_max_normalize(feature_vals_original)
        # 提取对应的SHAP值
        shap_vals = shap_values[:, i]
        
        # 构造图例标签（包含原始取值范围）
        label = f"TOP{i+1}: {feature_names[i]}\n(原始范围: {min_val:.1f} ~ {max_val:.1f})"
        
        # 绘制归一化后的散点图
        plt.scatter(
            feature_vals_norm, 
            shap_vals, 
            color=colors[i],
            marker=markers[i],
            alpha=0.6,
            s=30,
            label=label
        )
        
        # 添加趋势线（基于归一化后的横坐标）
        try:
            from scipy import stats
            # 计算线性回归趋势
            slope, intercept, r_value, p_value, std_err = stats.linregress(feature_vals_norm, shap_vals)
            # 生成趋势线x值（归一化后0~1）
            x_trend = np.linspace(0, 1, 100)
            y_trend = slope * x_trend + intercept
            # 绘制趋势线
            plt.plot(x_trend, y_trend, color=colors[i], linewidth=2, alpha=0.8)
        except:
            # 若线性回归失败，跳过趋势线
            pass
    
    # 设置图表样式
    plt.xlabel("Feature Value (Normalized to [0,1])", fontsize=14)
    plt.ylabel("SHAP Value", fontsize=14)
    plt.title(f"Combined SHAP Dependence Plot (Normalized X-axis) - {agent_name}", fontsize=16, pad=20)
    # 优化图例（避免重叠）
    plt.legend(fontsize=10, loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(labelsize=12)
    # 设置横坐标范围为0~1
    plt.xlim(-0.05, 1.05)
    
    # 保存图表 - 只保留SVG格式，注释掉PNG
    dep_svg = os.path.join(save_dir, f"{agent_name}_SHAP_Combined_Dependence_Normalized.svg")
    # dep_png = os.path.join(save_dir, f"{agent_name}_SHAP_Combined_Dependence_Normalized.png")
    plt.savefig(dep_svg, bbox_inches='tight', dpi=1200)
    # plt.savefig(dep_png, bbox_inches='tight', dpi=1200)
    print(f"✅ {agent_name} 归一化整合Dependence Plot已保存：{dep_svg}")
    
    if show_plot:
        plt.show()
    plt.close()

def shap_analysis_agent(agent, state_samples, feature_names, save_dir, agent_name, show_plot=False):
    """
    对单个智能体进行SHAP分析并绘制可视化图表（归一化横坐标的整合Dependence Plot）
    优化：仅保存绘图所需的全局统计信息，不保存原始大数组
    """
    # 1. 初始化SHAP解释器（使用KernelExplainer，适配任意模型）
    # 选择前100个样本作为背景集（加速计算）
    background_samples = state_samples[:100]
    explainer = shap.KernelExplainer(agent.predict_max_q, background_samples)
    
    # 2. 计算SHAP值（对所有采样样本）
    print(f"\n📊 正在计算{agent_name}的SHAP值（共{len(state_samples)}个样本）...")
    # 优化：nsamples=50平衡计算速度和准确性
    shap_values = explainer.shap_values(state_samples, nsamples=50)
    
    # ========== 优化：仅保存绘图所需的全局统计信息 ==========
    # 1. 状态样本的全局统计（均值、最值、标准差）- 表征数据分布
    state_stats = {
        feature_names[i]: {
            "mean": float(np.mean(state_samples[:, i])),
            "min": float(np.min(state_samples[:, i])),
            "max": float(np.max(state_samples[:, i])),
            "std": float(np.std(state_samples[:, i]))
        } for i in range(len(feature_names))
    }
    
    # 2. SHAP值的全局统计（均值、最值、标准差）- 表征特征影响分布
    shap_stats = {
        feature_names[i]: {
            "mean_shap_value": float(np.mean(shap_values[:, i])),
            "min_shap_value": float(np.min(shap_values[:, i])),
            "max_shap_value": float(np.max(shap_values[:, i])),
            "std_shap_value": float(np.std(shap_values[:, i])),
            "abs_mean_shap_value": float(np.mean(np.abs(shap_values[:, i])))  # 特征重要性
        } for i in range(len(feature_names))
    }
    
    # 3. TOP3特征的关键信息（绘图核心）
    feature_importance = np.abs(shap_values).mean(axis=0)
    top3_indices = np.argsort(-feature_importance)[:3]
    top3_features = []
    for idx in top3_indices:
        # 计算该特征的归一化参数（绘图用）
        feature_vals_norm, min_val, max_val = min_max_normalize(state_samples[:, idx])
        # 计算该特征SHAP值的线性回归参数（趋势线用）
        slope, intercept, r_value = np.nan, np.nan, np.nan
        try:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(feature_vals_norm, shap_values[:, idx])
        except:
            pass
        
        top3_features.append({
            "feature_name": feature_names[idx],
            "importance": float(feature_importance[idx]),
            "rank": int(np.where(top3_indices == idx)[0][0] + 1),
            "original_range": f"{min_val:.1f} ~ {max_val:.1f}",
            "regression_slope": float(slope),
            "regression_intercept": float(intercept),
            "r_squared": float(r_value**2) if not np.isnan(r_value) else np.nan
        })
    
    # 4. 汇总智能体的核心SHAP数据
    shap_core_data = {
        "agent_name": agent_name,
        "n_samples": len(state_samples),
        "expected_value": float(explainer.expected_value),  # SHAP基准值
        "state_statistics": state_stats,  # 状态样本统计
        "shap_statistics": shap_stats,    # SHAP值统计
        "top3_features": top3_features,   # TOP3特征（绘图核心）
        "feature_importance_ranking": [  # 所有特征重要性排名
            {
                "feature_name": feature_names[i],
                "importance": float(feature_importance[i]),
                "rank": int(np.argsort(np.argsort(-feature_importance))[i] + 1)
            } for i in range(len(feature_names))
        ]
    }
    
    # 保存单个智能体的核心数据JSON
    shap_json_path = os.path.join(save_dir, f"{agent_name}_SHAP_Core_Data.json")
    with open(shap_json_path, 'w', encoding='utf-8') as f:
        json.dump(shap_core_data, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    print(f"✅ {agent_name} SHAP核心数据已保存为JSON：{shap_json_path}")
    
    # 3. 绘制SHAP Summary Plot（核心：所有特征的影响汇总）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        features=state_samples,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        cmap=plt.get_cmap("RdYlBu_r"),
        plot_size=(12, 8)
    )
    plt.title(f"SHAP Summary Plot - {agent_name}", fontsize=16, pad=20)
    # 保存Summary Plot - 只保留SVG格式，注释掉PNG
    summary_svg = os.path.join(save_dir, f"{agent_name}_SHAP_Summary.svg")
    # summary_png = os.path.join(save_dir, f"{agent_name}_SHAP_Summary.png")
    plt.savefig(summary_svg, bbox_inches='tight', dpi=1200)
    # plt.savefig(summary_png, bbox_inches='tight', dpi=1200)
    print(f"✅ {agent_name} SHAP Summary Plot已保存：{summary_svg}")
    if show_plot:
        plt.show()
    plt.close()
    
    # 4. 绘制SHAP Force Plot（单个样本的详细影响，数字保留整数）
    sample_idx = 0
    # 生成Force Plot（先不显示）
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        features=state_samples[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        figsize=(15, 4),
        show=False  # 关键：先不显示，修改文本后再保存
    )
    
    # 核心修改：遍历所有文本元素，将小数转为整数
    for text in plt.gca().texts:
        text_str = text.get_text()
        # 正则匹配所有带小数点的数字（包括正负）
        nums = re.findall(r'-?\d+\.\d+', text_str)
        for num in nums:
            # 四舍五入转为整数
            int_num = str(round(float(num)))
            # 替换原文本中的小数为整数
            text_str = text_str.replace(num, int_num)
        # 更新文本内容
        text.set_text(text_str)
    
    plt.title(f"SHAP Force Plot - {agent_name} (Sample {sample_idx})", fontsize=14, pad=10)
    # 保存Force Plot - 只保留SVG格式，注释掉PNG
    force_svg = os.path.join(save_dir, f"{agent_name}_SHAP_Force.svg")
    # force_png = os.path.join(save_dir, f"{agent_name}_SHAP_Force.png")
    plt.savefig(force_svg, bbox_inches='tight', dpi=1200)
    # plt.savefig(force_png, bbox_inches='tight', dpi=1200)
    print(f"✅ {agent_name} SHAP Force Plot已保存：{force_svg}")
    if show_plot:
        plt.show()
    plt.close()
    
    # 5. 绘制归一化横坐标的整合Dependence Plot（TOP3特征）
    plot_combined_dependence_normalized(
        shap_values=shap_values,
        state_samples=state_samples,
        feature_names=feature_names,
        top_k=3,
        agent_name=agent_name,
        save_dir=save_dir,
        show_plot=show_plot
    )
    
    # 6. 计算特征重要性并保存
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': feature_importance,
        'Importance_Rank': np.argsort(np.argsort(-feature_importance)) + 1  # 排名
    }).sort_values('SHAP_Importance', ascending=False)
    
    # 保存特征重要性CSV
    importance_csv = os.path.join(save_dir, f"{agent_name}_SHAP_Importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding='utf-8')
    print(f"✅ {agent_name} 特征重要性已保存：{importance_csv}")
    
    # 返回核心数据，用于汇总三个智能体
    return importance_df, shap_core_data

# ====================== 6. 主程序 ======================
if __name__ == '__main__':
    args = parse_args()
    
    # 打印配置信息
    print("=" * 80)
    print("                    智能体SHAP分析配置（归一化横坐标的整合Dependence Plot）                  ")
    print("=" * 80)
    print(f"模型路径配置:")
    print(f"  - 日期文件夹: {args.net_date}")
    print(f"  - 训练ID: {args.train_id}")
    print(f"  - 模型前缀: {args.model_prefix}")
    print(f"  - RNN模型路径: {args.rnn_path}")
    print(f"SHAP配置:")
    print(f"  - 采样数量: {args.n_samples}")
    print(f"  - 显示图表: {'是' if args.show_plot else '否'}")
    print(f"  - Force图数字格式: 仅保留整数（四舍五入）")
    print(f"  - Dependence Plot: 归一化横坐标(0~1)的整合TOP3特征图")
    print(f"  - 图片保存格式: 仅SVG（PNG已注释）")
    print(f"  - 数据保存优化: 仅保存全局统计信息（均值/最值/标准差），不保存原始大数组")
    print(f"  - 数据输出: 单个智能体JSON + 三个智能体汇总JSON")
    print("状态维度定义（匹配Envs环境）:")
    feature_names = [
        'Load_Power (W)', 
        'Temperature (°C)', 
        'FC_Power (W)', 
        'Battery_Power (W)', 
        'SC_Power (W)', 
        'Battery_SOC', 
        'SC_SOC'
    ]
    for i, name in enumerate(feature_names):
        print(f"  - 维度{i}: {name}")
    print("=" * 80 + "\n")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化真实环境（使用你提供的Envs类）
    env = Envs()
    print(f"✅ 环境初始化完成，观测空间范围:")
    print(f"  - 下限: {env.observation_space.low}")
    print(f"  - 上限: {env.observation_space.high}")
    
    # 加载RNN模型
    try:
        rnn_model = MultiTaskRNN().to(device)
        rnn_model.load_state_dict(torch.load(args.rnn_path, map_location=device))
        rnn_model.eval()
        print(f"\n✅ 成功加载RNN模型: {args.rnn_path}")
    except FileNotFoundError as e:
        print(f"❌ RNN模型文件未找到: {e}")
        raise
    except Exception as e:
        print(f"❌ RNN模型加载失败: {e}")
        raise

    # 初始化三个智能体（匹配环境的动作空间）
    N_FC_ACTIONS = env.N_FC_ACTIONS  # 32
    N_BAT_ACTIONS = env.N_BAT_ACTIONS  # 40
    N_SC_ACTIONS = env.N_SC_ACTIONS  # 2
    
    FC_Agent = JointDQN("FC_Agent", rnn_model, N_FC_ACTIONS)
    Bat_Agent = JointDQN("Bat_Agent", rnn_model, N_BAT_ACTIONS)
    SC_Agent = JointDQN("SC_Agent", rnn_model, N_SC_ACTIONS)

    # 路径设置
    MODEL_BASE_DIR = os.path.join(project_root, "nets", "Chap4", "Joint_Net", args.net_date, args.train_id)
    SHAP_DIR = os.path.join(MODEL_BASE_DIR, "SHAP_Analysis")
    MODEL_FILE_PREFIX = os.path.join(MODEL_BASE_DIR, args.model_prefix)
    
    # 创建保存目录
    os.makedirs(SHAP_DIR, exist_ok=True)

    # 加载智能体权重
    try:
        FC_Agent.load_net(f"{MODEL_FILE_PREFIX}_FC.pth")
        Bat_Agent.load_net(f"{MODEL_FILE_PREFIX}_BAT.pth")
        SC_Agent.load_net(f"{MODEL_FILE_PREFIX}_SC.pth")
        print(f"\n✅ 成功加载所有智能体模型:")
        print(f"   模型路径: {MODEL_FILE_PREFIX}_*.pth")
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        print(f"   期望路径: {MODEL_FILE_PREFIX}_*.pth")
        raise
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

    # 设置智能体为评估模式
    FC_Agent.eval_net.eval()
    Bat_Agent.eval_net.eval()
    SC_Agent.eval_net.eval()

    # 生成状态采样数据集（严格匹配环境参数）
    print(f"\n📊 正在生成{args.n_samples}个状态采样样本（匹配真实环境）...")
    state_samples = generate_state_samples(env, n_samples=args.n_samples)
    print(f"✅ 状态采样完成，样本形状: {state_samples.shape}")

    # ====================== 逐个智能体进行SHAP分析 ======================
    agents = [FC_Agent, Bat_Agent, SC_Agent]
    all_importance = []
    all_shap_core_data = {}  # 存储三个智能体的核心数据，用于汇总
    
    for agent in agents:
        print(f"\n{'='*60}")
        print(f"开始分析 {agent.agent_name}")
        print(f"{'='*60}")
        
        # 单个智能体SHAP分析（返回核心数据）
        importance_df, shap_core_data = shap_analysis_agent(
            agent=agent,
            state_samples=state_samples,
            feature_names=feature_names,
            save_dir=SHAP_DIR,
            agent_name=agent.agent_name,
            show_plot=args.show_plot
        )
        
        # 收集数据
        importance_df['Agent'] = agent.agent_name
        all_importance.append(importance_df)
        all_shap_core_data[agent.agent_name] = shap_core_data
    
    # ========== 新增：保存三个智能体的汇总JSON文件 ==========
    combined_shap_data = {
        "analysis_config": {
            "net_date": args.net_date,
            "train_id": args.train_id,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "feature_names": feature_names,
            "analysis_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "agents": all_shap_core_data,
        "cross_agent_summary": {
            # 跨智能体的特征重要性对比（每个特征在不同智能体中的平均重要性）
            "feature_importance_cross_agent": [
                {
                    "feature_name": fname,
                    "FC_Agent_importance": all_shap_core_data["FC_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"],
                    "Bat_Agent_importance": all_shap_core_data["Bat_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"],
                    "SC_Agent_importance": all_shap_core_data["SC_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"],
                    "average_importance": float(np.mean([
                        all_shap_core_data["FC_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"],
                        all_shap_core_data["Bat_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"],
                        all_shap_core_data["SC_Agent"]["shap_statistics"][fname]["abs_mean_shap_value"]
                    ]))
                } for fname in feature_names
            ],
            # 各智能体TOP1特征汇总
            "top1_features_summary": [
                {
                    "agent_name": agent_name,
                    "top1_feature": all_shap_core_data[agent_name]["top3_features"][0]["feature_name"],
                    "top1_importance": all_shap_core_data[agent_name]["top3_features"][0]["importance"],
                    "top1_r_squared": all_shap_core_data[agent_name]["top3_features"][0]["r_squared"]
                } for agent_name in ["FC_Agent", "Bat_Agent", "SC_Agent"]
            ]
        }
    }
    
    # 保存汇总JSON文件
    combined_json_path = os.path.join(SHAP_DIR, "All_Agents_SHAP_Core_Data.json")
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_shap_data, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    print(f"\n✅ 三个智能体SHAP汇总数据已保存：{combined_json_path}")
    
    # 合并所有智能体的特征重要性并保存
    combined_importance = pd.concat(all_importance, ignore_index=True)
    combined_csv = os.path.join(SHAP_DIR, "All_Agents_SHAP_Importance.csv")
    combined_importance.to_csv(combined_csv, index=False, encoding='utf-8')
    
    # ====================== 完成提示 ======================
    print("\n" + "="*80)
    print("🎉 所有智能体SHAP分析完成！")
    print(f"📁 分析结果保存目录: {SHAP_DIR}")
    print(f"📋 生成的文件类型:")
    print(f"   1. SHAP_Summary.svg (特征影响汇总图，仅SVG格式)")
    print(f"   2. SHAP_Force.svg (单个样本详细影响图，数字仅保留整数，仅SVG格式)")
    print(f"   3. SHAP_Combined_Dependence_Normalized.svg (归一化横坐标的整合依赖图，仅SVG格式)")
    print(f"   4. *_SHAP_Importance.csv (特征重要性量化表)")
    print(f"   5. All_Agents_SHAP_Importance.csv (所有智能体特征重要性汇总)")
    print(f"   6. *_SHAP_Core_Data.json (单个智能体核心数据，仅含全局统计信息)")
    print(f"   7. All_Agents_SHAP_Core_Data.json (三个智能体汇总数据，含跨智能体对比)")
    print("="*80)
    
    # 打印特征重要性汇总
    print("\n📊 各智能体TOP3重要特征:")
    for agent_name in ["FC_Agent", "Bat_Agent", "SC_Agent"]:
        agent_importance = combined_importance[combined_importance['Agent'] == agent_name].head(3)
        print(f"\n{agent_name}:")
        for _, row in agent_importance.iterrows():
            print(f"  - {row['Feature']} (重要性: {row['SHAP_Importance']:.4f}, 排名: {row['Importance_Rank']})")