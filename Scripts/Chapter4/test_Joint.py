import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import os
import json
import argparse
import sys
from json import JSONEncoder
import torch.nn as nn
import torch.nn.functional as F

# ====================== 1. 环境与路径配置 ======================
def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设脚本在 Scripts/Chapter4/，向上两级到项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# 导入原有引擎组件
from Scripts.Chapter3.MARL_Engine import Net, IndependentDQN, device
from Scripts.Env import Envs
from Scripts.utils.global_utils import font_get

# 获取字体设置
font_get()

# ====================== 2. JointNet 相关类定义 ======================
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

    def choose_action(self, x, train=False, epsilon=0.9):
        x_tensor = torch.FloatTensor(x).to(device)
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        
        if train and np.random.uniform() >= epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                actions_value = self.eval_net(x_tensor)
            return torch.max(actions_value, 1)[1].item()

# ====================== 3. 工具类与参数解析 ======================
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray, torch.Tensor)):
            if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
            return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
        return super(NumpyEncoder, self).default(obj)

def parse_args():
    parser = argparse.ArgumentParser(description='JointNet 测试脚本')
    parser.add_argument('--net-date', type=str, required=True, help='JointNet 日期文件夹 (如 1219)')
    parser.add_argument('--train-id', type=str, required=True, help='训练 ID')
    parser.add_argument('--rnn-path', type=str, default=os.path.join(project_root, "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth"))
    parser.add_argument('--model-prefix', type=str, default="Joint_Model")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-time', type=float, default=800.0)
    parser.add_argument('--sc-threshold', type=float, default=1e-3)
    parser.add_argument('--show-plot', action='store_true')
    return parser.parse_args()

# ====================== 4. 主测试程序 ======================
if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 初始化环境
    env = Envs()
    dt = getattr(env, "dt", 1.0)
    loads = env.loads
    temperature = env.temperature
    
    # 加载 RNN
    rnn_model = MultiTaskRNN().to(device)
    rnn_model.load_state_dict(torch.load(args.rnn_path, map_location=device))
    rnn_model.eval()

    # 初始化智能体
    FC_Agent = JointDQN("FC_Agent", rnn_model, 32)
    Bat_Agent = JointDQN("Bat_Agent", rnn_model, 40)
    SC_Agent = JointDQN("SC_Agent", rnn_model, 2)

    # 路径设置
    MODEL_BASE_DIR = os.path.join(project_root, "nets", "Chap4", "Joint_Net", args.net_date, args.train_id)
    SAVE_DIR = MODEL_BASE_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载权重
    try:
        FC_Agent.load_net(os.path.join(MODEL_BASE_DIR, f"{args.model_prefix}_FC.pth"))
        Bat_Agent.load_net(os.path.join(MODEL_BASE_DIR, f"{args.model_prefix}_BAT.pth"))
        SC_Agent.load_net(os.path.join(MODEL_BASE_DIR, f"{args.model_prefix}_SC.pth"))
        print(f"✅ Models loaded from {MODEL_BASE_DIR}")
    except Exception as e:
        print(f"❌ Load error: {e}"); exit()

    # --- 循环变量初始化 ---
    s = env.reset()
    step = 0
    power_fc, battery_power, power_sc = [], [], []
    soc_bat, soc_sc_list, times, unmatched_power_list = [], [], [], []
    ep_r, total_fc_H2, total_bat_H2 = 0, 0, 0
    sc_inactive_steps = 0
    bat_charge_steps = 0
    total_unmatched_power = 0
    total_unmatched_energy = 0
    time_start = time.time()

    print("🚀 测试运行中...")
    while True:
        # 动作选择 (7维输入)
        a_fc = FC_Agent.choose_action(s, train=False)
        a_bat = Bat_Agent.choose_action(s, train=False)
        a_sc = SC_Agent.choose_action(s, train=False)

        s_, r, done, info = env.step([a_fc, a_bat, a_sc])

        # 记录数据
        times.append(step * dt)
        cur_fc, cur_bat, cur_sc = float(s_[2]), float(s_[3]), float(s_[4])
        power_fc.append(cur_fc)
        battery_power.append(cur_bat)
        power_sc.append(cur_sc)
        soc_bat.append(float(s_[5]))
        soc_sc_list.append(float(s_[6]))

        # 未匹配功率计算
        if step < len(loads):
            unmatch = loads[step] - (cur_fc + cur_bat + cur_sc)
            unmatched_power_list.append(unmatch)
            total_unmatched_power += abs(unmatch)
            total_unmatched_energy += abs(unmatch) * dt / 3600.0

        if abs(cur_sc) < args.sc_threshold: sc_inactive_steps += 1
        if cur_bat < 0: bat_charge_steps += 1
        
        total_fc_H2 += float(info.get("C_fc_g", 0))
        total_bat_H2 += float(info.get("C_bat_g", 0))
        ep_r += r

        if done or step * dt >= args.max_time - dt: break
        s = s_
        step += 1

    total_steps = len(times)
    total_time_cost = time.time() - time_start

    # ====================== 5. 绘图部分 ======================
    
    plt.rcParams.update({'font.family': ['Times New Roman'], 'font.size': 12})
    LABEL_FONT_SIZE = 18
    colors = ['#f09639', '#c84343', '#42985e', '#8a7ab5', '#3570a8']
    
    plot_times = times[:len(power_fc)]
    plot_loads = loads[:len(power_fc)]
    
    # --- 总图绘制 ---
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(plot_times, plot_loads, label='Power Demand', color=colors[4])
    ax1.plot(plot_times, power_fc, label='Power Fuel Cell', color=colors[0])
    ax1.plot(plot_times, battery_power, label='Power Battery', color=colors[2])
    ax1.plot(plot_times, power_sc, label='Power SuperCap', color='k', linestyle='--')
    ax1.set_ylabel('Power/W', fontsize=LABEL_FONT_SIZE)
    ax1.set_xlabel('Time/s', fontsize=LABEL_FONT_SIZE)
    ax1.set_xlim(0, args.max_time); ax1.set_ylim(-2500, 5500)

    ax2 = ax1.twinx()
    ax2.plot(plot_times, soc_bat, label='Battery SOC', color=colors[1])
    ax2.plot(plot_times, soc_sc_list, label='SuperCap SOC', color='grey', linestyle=':')
    ax2.set_ylabel('SOC', fontsize=LABEL_FONT_SIZE); ax2.set_ylim(0, 1)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.savefig(os.path.join(SAVE_DIR, f"{args.model_prefix}_Total_Test.png"), dpi=300, bbox_inches='tight')

    # --- 拆分图绘制 (multi_figures 子目录) ---
    multi_dir = os.path.join(SAVE_DIR, "multi_figures")
    os.makedirs(multi_dir, exist_ok=True)
    
    # 示例：FC 功率图
    fig_fc, ax_fc = plt.subplots(figsize=(15, 6))
    ax_fc.plot(plot_times, plot_loads, label='Demand', color='#3570a8')
    ax_fc.plot(plot_times, power_fc, label='FC Power', color='#f09639')
    ax_fc_temp = ax_fc.twinx()
    ax_fc_temp.plot(plot_times, temperature[:len(power_fc)], label='Temp', color='#8a7ab5')
    ax_fc_temp.set_ylim(-25, 40)
    fig_fc.savefig(os.path.join(multi_dir, "FC_Power_Detail.png"), dpi=300)
    plt.close(fig_fc)

    # ====================== 6. JSON 保存 ======================
    final_results = {
        "hydrogen": {"total": total_fc_H2 + total_bat_H2, "fc": total_fc_H2, "bat": total_bat_H2},
        "matching": {"avg_unmatch": total_unmatched_power/total_steps, "energy_unmatch_wh": total_unmatched_energy},
        "performance": {"reward": ep_r, "time_cost": total_time_cost}
    }
    with open(os.path.join(SAVE_DIR, "test_results.json"), 'w') as f:
        json.dump(final_results, f, cls=NumpyEncoder, indent=4)

    print(f"✅ 测试完成！结果保存至: {SAVE_DIR}")
    if args.show_plot: plt.show()