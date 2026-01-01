import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入Chapter3的相关组件
from Scripts.Chapter3.MARL_Engine import Net, IndependentDQN, device

# 导入Chapter5的环境和工具
from Scripts.Chapter5.Env_Ultra import EnvUltra
from Scripts.Chapter5.Meta_RL_Engine import ResultSaver, create_output_dir, RuleBasedPolicy, DPStrategy

# ====================== Joint_Net相关类定义 ======================
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型和其他非标准类型"""
    def default(self, obj):
        # 处理numpy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理torch张量
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        # 处理其他数值类型
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        # 调用父类默认方法处理其他类型
        return super(NumpyEncoder, self).default(obj)

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

# ====================== 1. 工具函数 ======================
def load_joint_net_models(rnn_path, joint_net_path_prefix):
    """
    加载Chapter4的Joint_Net模型
    
    Args:
        rnn_path: RNN模型路径
        joint_net_path_prefix: Joint_Net模型前缀
    
    Returns:
        三个智能体：FC_Agent, Bat_Agent, SC_Agent
    """
    # 加载RNN模型
    rnn_model = MultiTaskRNN().to(device)
    rnn_model.load_state_dict(torch.load(rnn_path, map_location=device))
    rnn_model.eval()
    print(f"✅ 成功加载RNN模型: {rnn_path}")
    
    # 定义动作空间大小（与Chapter4一致）
    N_FC_ACTIONS = 32
    N_BAT_ACTIONS = 40
    N_SC_ACTIONS = 2
    
    # 初始化智能体
    FC_Agent = JointDQN("FC_Agent", rnn_model, N_FC_ACTIONS)
    Bat_Agent = JointDQN("Bat_Agent", rnn_model, N_BAT_ACTIONS)
    SC_Agent = JointDQN("SC_Agent", rnn_model, N_SC_ACTIONS)
    
    # 加载权重
    FC_Agent.load_net(f"{joint_net_path_prefix}_FC.pth")
    Bat_Agent.load_net(f"{joint_net_path_prefix}_BAT.pth")
    SC_Agent.load_net(f"{joint_net_path_prefix}_SC.pth")
    
    print(f"✅ 成功加载JointNet模型: {joint_net_path_prefix}_*.pth")
    
    return FC_Agent, Bat_Agent, SC_Agent

# ====================== 2. 测试函数 ======================
def test_algorithm(algorithm_name, agent_list, env, max_steps=1000):
    """
    测试指定算法在环境中的表现
    
    Args:
        algorithm_name: 算法名称
        agent_list: 智能体列表或策略对象
        env: 测试环境
        max_steps: 最大测试步数
    
    Returns:
        测试结果，包括功率分配数据和性能指标
    """
    state = env.reset()
    
    # 初始化数据存储
    power_fc = []
    power_bat = []
    power_sc = []
    load_power = []
    soc_bat = []
    soc_sc = []
    rewards = []
    
    total_reward = 0.0
    steps = 0
    
    while steps < max_steps:
        # 根据算法类型选择动作
        if algorithm_name == "Joint_Net":
            # 使用Joint_Net智能体
            a_fc = agent_list[0].choose_action(state, train=False)
            a_bat = agent_list[1].choose_action(state, train=False)
            a_sc = agent_list[2].choose_action(state, train=False)
        elif algorithm_name in ["Rule_Based", "DP"]:
            # 使用基于规则或DP的策略
            a_fc, a_bat, a_sc = agent_list.choose_action(state)
        else:
            # 其他算法
            raise ValueError(f"未知算法: {algorithm_name}")
        
        action_list = [a_fc, a_bat, a_sc]
        
        # 执行动作
        next_state, reward, done, info = env.step(action_list)
        
        # 记录数据
        power_fc.append(float(next_state[2]))
        power_bat.append(float(next_state[3]))
        power_sc.append(float(next_state[4]))
        load_power.append(float(state[0]))  # 负载功率是当前状态的第一个元素
        soc_bat.append(float(next_state[5]))
        soc_sc.append(float(next_state[6]))
        rewards.append(reward)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if done:
            break
    
    # 计算性能指标
    avg_reward = np.mean(rewards) if rewards else 0.0
    std_reward = np.std(rewards) if rewards else 0.0
    
    # 准备功率分配数据
    power_data = {
        'power_fc': power_fc,
        'power_bat': power_bat,
        'power_sc': power_sc,
        'load_power': load_power,
        'soc_bat': soc_bat,
        'soc_sc': soc_sc,
        'temperature': [env.temperature] * steps  # 假设环境温度恒定
    }
    
    # 准备性能数据
    performance = {
        'algorithm': algorithm_name,
        'total_steps': steps,
        'total_reward': total_reward,
        'average_reward': avg_reward,
        'std_reward': std_reward,
        'power_fc_avg': np.mean(power_fc) if power_fc else 0.0,
        'power_bat_avg': np.mean(power_bat) if power_bat else 0.0,
        'power_sc_avg': np.mean(power_sc) if power_sc else 0.0,
        'soc_bat_min': np.min(soc_bat) if soc_bat else 0.0,
        'soc_bat_max': np.max(soc_bat) if soc_bat else 0.0,
        'soc_sc_min': np.min(soc_sc) if soc_sc else 0.0,
        'soc_sc_max': np.max(soc_sc) if soc_sc else 0.0
    }
    
    return power_data, performance

# ====================== 3. 主函数 ======================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Chapter5对比实验脚本')
    parser.add_argument('--rnn-path', type=str, 
                        default='/home/siyu/Master_Code/nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth',
                        help='预训练RNN模型路径')
    parser.add_argument('--joint-net-path-prefix', type=str, 
                        default='/home/siyu/Master_Code/nets/Chap4/Joint_Net/1223/3/Joint_Model',
                        help='Joint_Net模型前缀')
    parser.add_argument('--max-steps', type=int, default=1000, help='最大测试步数')
    parser.add_argument('--output-dir', type=str, default='', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    if not args.output_dir:
        output_dir = create_output_dir("comparison_experiments")
    else:
        output_dir = args.output_dir
    
    # 初始化结果保存器
    result_saver = ResultSaver(output_dir)
    
    # 加载Joint_Net模型
    fc_agent, bat_agent, sc_agent = load_joint_net_models(args.rnn_path, args.joint_net_path_prefix)
    
    # 初始化算法列表
    algorithms = {
        "Joint_Net": [fc_agent, bat_agent, sc_agent],
        "Rule_Based": RuleBasedPolicy(),
        "DP": DPStrategy()
    }
    
    # 定义测试场景（Chapter5的三个典型工况）
    test_scenarios = [
        "air",           # 空中飞行工况
        "surface",       # 水面航行工况
        "underwater"     # 水下潜航工况
    ]
    
    # 所有场景的测试结果
    all_scenarios_results = {
        "config": {
            "rnn_path": args.rnn_path,
            "joint_net_path_prefix": args.joint_net_path_prefix,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scenarios": {}
    }
    
    # 对每个场景进行测试
    for scenario in test_scenarios:
        print(f"\n=== 开始测试场景: {scenario} ===")
        
        # 创建环境
        env = EnvUltra(scenario_type=scenario)
        
        # 场景测试结果
        scenario_results = {}
        
        # 测试每种算法
        for algo_name, algo_agent in algorithms.items():
            print(f"\n🔍 测试算法: {algo_name}")
            
            # 测试算法
            power_data, performance = test_algorithm(algo_name, algo_agent, env, max_steps=args.max_steps)
            
            # 保存功率分配图
            plot_filename = f"power_distribution_{scenario}_{algo_name}.svg"
            result_saver.save_power_distribution_plot(power_data, scenario, filename=plot_filename)
            
            # 保存性能数据
            scenario_results[algo_name] = {
                "performance": performance,
                "power_data": power_data
            }
            
            print(f"✅ 算法 '{algo_name}' 测试完成")
            print(f"   总奖励: {performance['total_reward']:.4f}")
            print(f"   平均奖励: {performance['average_reward']:.4f}")
        
        # 保存场景结果
        all_scenarios_results["scenarios"][scenario] = scenario_results
    
    # 保存所有测试结果
    result_saver.save_results_json(all_scenarios_results, "comparison_test_results.json")
    
    print(f"\n=== 所有测试完成 ===")
    print(f"所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
