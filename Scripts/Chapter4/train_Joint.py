import os
import time
import json
import subprocess
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch

def setup_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root

project_root = setup_path()

# ----------------------------------------------------
# 1. 环境与路径配置
# ----------------------------------------------------
from Scripts.Chapter3.MARL_Engine import (
    setup_project_root, device, 
    get_max_folder_name
)
project_root = setup_project_root()
from Scripts.Env import Envs
from Scripts.utils.global_utils import *

# 确保 Joint_Net 定义正确
from Scripts.Chapter4.Joint_Net import MultiTaskRNN, JointNet, JointDQN

font_get()

"""
示例训练脚本

从头训练

nohup python Scripts/Chapter4/train_Joint.py \
--episode 5000 \
--pool-size 200 > logs/1222_5.log 2>&1 &

继续训练

nohup python Scripts/Chapter4/train_Joint.py \
--resume-training \
--pretrain-date 1223 \
--pretrain-train-id 0 \
--epsilon 0.9 \
--lr 1e-5 \
--pretrain-model-prefix "Joint_Model" \
--episode 2000 > logs/1223_3.log 2>&1 &

"""

# ====================== 新增：命令行参数解析（对齐train.py） ======================
def parse_args():
    """解析命令行参数（支持从头训练/继续训练）"""
    parser = argparse.ArgumentParser(description='JointNet训练脚本（支持从头训练/继续训练）')
    
    # 核心训练模式参数
    parser.add_argument('--resume-training', action='store_true', 
                        help='是否基于已有模型继续训练（默认：从头训练）')
    parser.add_argument('--pretrain-date', type=str, default="1219",
                        help='预训练模型的日期文件夹（仅resume-training=True时生效）')
    parser.add_argument('--pretrain-train-id', type=str, default="1",
                        help='预训练模型的train_id（仅resume-training=True时生效）')
    parser.add_argument('--pretrain-model-prefix', type=str, 
                        default="Joint_Model",
                        help='预训练模型前缀（仅resume-training=True时生效）')
    
    # 继续训练示例代码：--resume-training --pretrain-date 1219 --pretrain-train-id 5

    # 训练超参数（可选，支持命令行覆盖默认值）
    parser.add_argument('--batch-size', type=int, default=32, help='批大小（默认：32）')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率（默认：1e-5）')
    parser.add_argument('--epsilon', type=float, default=0.9, help='探索率（默认：0.9）')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子（默认：0.95）')
    parser.add_argument('--pool-size', type=int, default=100, help='池大小（默认：20）')
    parser.add_argument('--episode', type=int, default=2000, help='训练回合数（默认：2000）')
    parser.add_argument('--learn-frequency', type=int, default=5, help='学习频率（默认：5）')
    parser.add_argument('--remark', type=str, default="", help='备注')
    
    # 路径参数（可选）
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard日志目录（默认：自动生成）')
    parser.add_argument('--init-rnn-path', type=str, 
                        default=os.path.join(project_root, "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/rnn_classifier_multitask.pth"),
                        help='从头训练时初始化RNN的路径（resume-training=True时无效）')
    
    return parser.parse_args()

args = parse_args()

# ====================== 全局配置（对齐train.py） ======================
try:
    env = Envs()
except Exception as e:
    print(f"❌ 环境初始化失败: {e}")
    sys.exit()

# 动态配置超参数（从命令行参数读取）
BATCH_SIZE = args.batch_size
LR = args.lr
EPSILON = args.epsilon
GAMMA = args.gamma
TARGET_REPLACE_ITER = 100
POOL_SIZE = args.pool_size
EPISODE = args.episode
LEARN_FREQUENCY = args.learn_frequency
REAL_TIME_DRAW = False

# 继续训练配置
RESUME_TRAINING = args.resume_training
PRETRAIN_DATE = args.pretrain_date
PRETRAIN_TRAIN_ID = args.pretrain_train_id
PRETRAIN_MODEL_PREFIX = args.pretrain_model_prefix
GLOBAL_SEED = 42

# 学习率调度与早停参数
LR_PATIENCE = 100
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 1000
REWARD_THRESHOLD = 0.001

torch.set_default_dtype(torch.float32)
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

N_STATES = 7  # JointNet固定7维输入
# 确保 MEMORY_CAPACITY 至少大于 batch_size
MEMORY_CAPACITY = max(env.step_length * POOL_SIZE, BATCH_SIZE * 2)
execute_date = time.strftime("%m%d", time.localtime())
execute_time = time.strftime("%H%M%S", time.localtime())  # 新增：记录具体时间

# 全局变量存储最优模型文件名
best_model_base_name = "Joint_Model"
remark = args.remark  # 初始化remark

# ====================== 新增：保存超参数函数（对齐train.py） ======================
def save_hyperparameters(save_path, final_metrics=None):
    """
    保存超参数到指定路径（txt和json格式）
    :param save_path: 保存目录
    :param final_metrics: 训练最终指标（如最大奖励、最终奖励等）
    """
    # 整理超参数字典
    hyperparams = {
        # 基础信息
        "train_info": {
            "execute_date": execute_date,
            "execute_time": execute_time,
            "train_id": os.path.basename(save_path),
            "remark": remark,
            "device": str(device),
            "seed": GLOBAL_SEED,
            "total_training_time_s": round(time.time() - start_time_total, 2) if 'start_time_total' in globals() else 0,
            "best_model_base_name": best_model_base_name,
            "best_model_full_path": os.path.join(save_path, best_model_base_name) if best_model_base_name else "",
            "resume_training": RESUME_TRAINING,
            "command_line_args": vars(args),
            "pretrain_model_info": {
                "pretrain_date": PRETRAIN_DATE if RESUME_TRAINING else "",
                "pretrain_train_id": PRETRAIN_TRAIN_ID if RESUME_TRAINING else "",
                "pretrain_model_prefix": PRETRAIN_MODEL_PREFIX if RESUME_TRAINING else "",
                "init_rnn_path": args.init_rnn_path if not RESUME_TRAINING else "NOT_USED"
            }
        },
        # 核心超参数
        "core_hyperparams": {
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "EPSILON": EPSILON,
            "GAMMA": GAMMA,
            "TARGET_REPLACE_ITER": TARGET_REPLACE_ITER,
            "POOL_SIZE": POOL_SIZE,
            "EPISODE": EPISODE,
            "LEARN_FREQUENCY": LEARN_FREQUENCY,
            "MEMORY_CAPACITY": MEMORY_CAPACITY,
            "REAL_TIME_DRAW": REAL_TIME_DRAW
        },
        # 学习率调度与早停参数
        "lr_earlystop_params": {
            "LR_PATIENCE": LR_PATIENCE,
            "LR_FACTOR": LR_FACTOR,
            "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
            "REWARD_THRESHOLD": REWARD_THRESHOLD
        },
        # 环境参数
        "env_params": {
            "N_STATES": N_STATES,
            "MEMORY_WIDTH": MEMORY_WIDTH if 'MEMORY_WIDTH' in globals() else 0,
            "step_length": env.step_length if hasattr(env, 'step_length') else "unknown"
        },
        # 训练结果指标
        "training_metrics": final_metrics or {}
    }

    # 保存为JSON格式
    json_path = os.path.join(save_path, "hyperparameters.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

    # 保存为TXT格式
    txt_path = os.path.join(save_path, "hyperparameters.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                JointNet训练超参数汇总                \n")
        f.write("=" * 80 + "\n\n")
        
        for section, params in hyperparams.items():
            f.write(f"【{section.upper()}】\n")
            f.write("-" * 60 + "\n")
            for key, value in params.items():
                if key in ["best_model_base_name", "best_model_full_path", "resume_training", "init_rnn_path"]:
                    f.write(f"{key:<30}: \033[1;32m{value}\033[0m\n")
                else:
                    f.write(f"{key:<30}: {value}\n")
            f.write("\n")

    print(f"\n✅ 超参数已保存到：")
    print(f"   JSON格式: {json_path}")
    print(f"   TXT格式: {txt_path}")

# ====================== 时间分解打印函数（对齐train.py） ======================
def print_time_breakdown(episode, episode_times):
    total_time = sum(episode_times.values())
    if total_time < 1e-6:
        print(f"回合 {episode} 耗时过短，跳过耗时分析。")
        return

    print("\n" + "=" * 45)
    print(f"🚀 回合 {episode} 耗时分解 (总耗时: {total_time:.4f} s)")
    print("-" * 45)
    for name, time_val in sorted(episode_times.items(), key=lambda x: x[1], reverse=True):
        percentage = (time_val / total_time) * 100
        print(f"| {name.ljust(15)} | {time_val:9.4f} s | {percentage:6.2f} % |")
    print("=" * 45)

# ====================== 加载完整Joint模型函数（核心修改） ======================
def load_full_joint_agents(pretrain_date, pretrain_id, prefix):
    """加载包含RNN的完整JointNet智能体"""
    pretrain_dir = os.path.join(project_root, "nets", "Chap4", "Joint_Net", pretrain_date, pretrain_id)
    agents = {}
    names = ["FC", "BAT", "SC"]
    rnn_model = None
    
    if not os.path.exists(pretrain_dir):
        raise FileNotFoundError(f"预训练目录不存在: {pretrain_dir}")

    # 第一步：检查所有模型文件
    missing_agent_names = []
    existing_paths = {}
    
    for name in names:
        path = os.path.join(pretrain_dir, f"{prefix}_{name}.pth")
        if not os.path.exists(path):
            path = os.path.join(pretrain_dir, f"Joint_Model_{name}.pth")
            if not os.path.exists(path):
                missing_agent_names.append(name)
                continue
        existing_paths[name] = path

    # 第二步：处理缺失模型
    if missing_agent_names:
        print("\n❌ 以下JointNet模型文件未找到：")
        for name in missing_agent_names:
            print(f"   - {name} Agent")
        
        # 交互确认是否重新初始化
        while True:
            user_input = input("\n📌 是否重新初始化这些缺失的智能体？(y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                # 重新初始化缺失的智能体（需要先初始化RNN）
                print("\n🔄 重新初始化RNN模型（使用默认初始路径）...")
                rnn_model = MultiTaskRNN().to(device)
                rnn_model.load_state_dict(torch.load(args.init_rnn_path, map_location=device))
                rnn_model.train()
                
                action_dims = {"FC": 32, "BAT": 40, "SC": 2}  # 默认动作维度
                for name in missing_agent_names:
                    print(f"\n🔄 重新初始化{name} Agent（从0开始）...")
                    agent = JointDQN(name, rnn_model, action_dims[name])
                    agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
                    agents[name] = agent
                    print(f"✅ {name} Agent已重新初始化完成")
                break
            elif user_input in ['n', 'no']:
                print("\n🛑 用户选择终止训练，退出程序...")
                sys.exit(0)
            else:
                print("⚠️ 输入无效，请输入 y/yes 或 n/no！")

    # 第三步：加载存在的完整Joint模型
    for name in existing_paths:
        path = existing_paths[name]
        try:
            ckpt = torch.load(path, map_location=device)
            
            # 判断是否是包含RNN的完整模型
            has_rnn_params = any(key.startswith('rnn_part.') for key in ckpt.keys())
            if has_rnn_params:
                print(f"\n📌 检测到{name}模型包含RNN参数，加载完整Joint模型...")
                # 初始化RNN（第一次加载时）
                if rnn_model is None:
                    rnn_model = MultiTaskRNN().to(device)
                
                # 获取动作维度
                try:
                    n_act = ckpt['marl_part.output.weight'].shape[0]
                except KeyError:
                    n_act = ckpt['output.weight'].shape[0]
                
                # 初始化Agent并加载完整参数（包含RNN）
                agent = JointDQN(name, rnn_model, n_act)
                agent.eval_net.load_state_dict(ckpt)
                agent.target_net.load_state_dict(ckpt)
                agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
                agents[name] = agent
                
                # 更新RNN模型（所有Agent共享同一个RNN）
                rnn_model = agent.eval_net.rnn_part
                rnn_model.train()
                
                print(f"✅ 成功加载包含RNN的{name}完整Joint模型: {path}")
            else:
                print(f"\n📌 {name}模型不包含RNN参数，加载传统MARL模型...")
                # 兼容旧模型，需要初始化RNN
                if rnn_model is None:
                    rnn_model = MultiTaskRNN().to(device)
                    rnn_model.load_state_dict(torch.load(args.init_rnn_path, map_location=device))
                    rnn_model.train()
                
                # 获取动作维度
                try:
                    n_act = ckpt['output.weight'].shape[0]
                except KeyError:
                    n_act = 32 if name == "FC" else 40 if name == "BAT" else 2
                
                agent = JointDQN(name, rnn_model, n_act)
                agent.eval_net.marl_part.load_state_dict(ckpt)
                agent.target_net.marl_part.load_state_dict(ckpt)
                agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
                agents[name] = agent
                print(f"✅ 成功加载{name} MARL模型（使用初始RNN）: {path}")
                
        except Exception as e:
            raise RuntimeError(f"加载{name} Agent失败: {e}")
    
    # 确保所有Agent共享同一个RNN模型
    if rnn_model is None:
        raise RuntimeError("未能初始化/加载RNN模型")
    
    for name in agents:
        agents[name].eval_net.rnn_part = rnn_model
        agents[name].target_net.rnn_part = rnn_model
    
    return agents["FC"], agents["BAT"], agents["SC"], rnn_model

# ====================== Main 训练逻辑（完整增强版） ======================
if __name__ == '__main__':
    # 打印配置确认信息
    print("=" * 80)
    print("                    JointNet训练配置确认                  ")
    print("=" * 80)
    print(f"训练模式: {'继续训练（基于已有模型）' if RESUME_TRAINING else '从头训练'}")
    if RESUME_TRAINING:
        print(f"预训练模型配置:")
        print(f"  - 日期文件夹: {PRETRAIN_DATE}")
        print(f"  - Train ID: {PRETRAIN_TRAIN_ID}")
        print(f"  - 模型前缀: {PRETRAIN_MODEL_PREFIX}")
        print(f"  - 初始RNN路径: 【继续训练模式，不使用】")
    else:
        print(f"  - 初始RNN路径: {args.init_rnn_path}")
    print(f"核心超参数:")
    print(f"  - 批大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LR:.6f}")
    print(f"  - 探索率: {EPSILON}")
    print(f"  - 训练回合数: {EPISODE}")
    print("=" * 80 + "\n")

    # 1. 准备目录
    TARGET_BASE_DIR = os.path.join(project_root, "nets", "Chap4", "Joint_Net", execute_date)
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    train_id = get_max_folder_name(TARGET_BASE_DIR)
    base_path = os.path.join(TARGET_BASE_DIR, str(train_id))
    os.makedirs(base_path, exist_ok=True)
    
    # 更新remark
    if RESUME_TRAINING:
        remark = f"RESUME_JOINT_{execute_date}_{train_id}"
    else:
        remark = f"JOINT_{execute_date}_{train_id}"

    # 2. TensorBoard日志
    log_dir = args.log_dir if args.log_dir else os.path.join(base_path, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    # 3. 初始化/加载模型（核心修改）
    rnn_model = None
    FC_Agent, Bat_Agent, SC_Agent = None, None, None
    
    if RESUME_TRAINING:
        print("\n📌 开始加载包含RNN的完整预训练JointNet模型...")
        try:
            FC_Agent, Bat_Agent, SC_Agent, rnn_model = load_full_joint_agents(
                PRETRAIN_DATE, PRETRAIN_TRAIN_ID, PRETRAIN_MODEL_PREFIX
            )
            print(f"✅ 成功加载所有包含RNN的完整JointNet智能体")
        except Exception as e:
            print(f"❌ 加载完整Joint模型失败: {e}")
            raise
    else:
        print("\n📌 从头初始化JointNet智能体（包含RNN）...")
        # 初始化基础 RNN
        try:
            rnn_model = MultiTaskRNN().to(device)
            rnn_model.load_state_dict(torch.load(args.init_rnn_path, map_location=device))
            rnn_model.train()  # 设置为训练模式，允许反向传播
        except FileNotFoundError as e:
            print(f"❌ 初始RNN模型文件未找到: {e}")
            raise
        except Exception as e:
            print(f"❌ 初始RNN模型加载失败: {e}")
            raise
        
        # 从头初始化智能体
        FC_Agent = JointDQN("FC", rnn_model, 32)
        Bat_Agent = JointDQN("BAT", rnn_model, 40)
        SC_Agent = JointDQN("SC", rnn_model, 2)
        # 设置优化器
        FC_Agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
        Bat_Agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
        SC_Agent.setup_optimizer(LR, LR_FACTOR, LR_PATIENCE)
        print(f"✅ 成功初始化所有JointNet智能体（包含RNN）")

    all_agents = [FC_Agent, Bat_Agent, SC_Agent]

    # 5. 共享内存初始化 (关键修复)
    MEMORY_WIDTH = N_STATES * 2 + 3 + 1  # s(7), a1, a2, a3, r(1), s_(7) = 18
    shared_memory = np.zeros((MEMORY_CAPACITY, MEMORY_WIDTH))
    memory_counter = [0]
    
    # 绑定共享内存到智能体
    for a in all_agents:
        a.shared_memory = shared_memory
        a.memory_counter = memory_counter

    # 6. 训练循环（完整增强版）
    print(f'\n🚀 JointNet训练开始 [ID: {train_id}] [Device: {device}]')
    start_time_total = time.time()
    reward_max = -float('inf')
    reward_not_improve_episodes = 0
    training_done = False
    x_episodes, y_rewards, loss_records = [], [], []

    if REAL_TIME_DRAW:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(x_episodes, y_rewards)

    episode_pbar = tqdm(range(EPISODE), desc=f"JointNet Training")
    for i_episode in episode_pbar:
        if training_done:
            break

        # 确保RNN处于训练模式
        if rnn_model is not None:
            rnn_model.train()
        s = env.reset()
        ep_r = 0
        episode_times = {
            'Action_Select': 0.0,
            'Env_Step': 0.0,
            'Data_Store': 0.0,
            'DQN_Learn': 0.0
        }
        step_count = 0
        step_loss = []

        while True:
            # 动作选择
            time_start_action = time.time()
            a_fc = FC_Agent.choose_action(s, train=True, epsilon=EPSILON)
            a_bat = Bat_Agent.choose_action(s, train=True, epsilon=EPSILON)
            a_sc = SC_Agent.choose_action(s, train=True, epsilon=EPSILON)
            episode_times['Action_Select'] += (time.time() - time_start_action)

            # 环境交互
            action_list = [a_fc, a_bat, a_sc]
            time_start_step = time.time()
            s_, r, done, _ = env.step(action_list)
            episode_times['Env_Step'] += (time.time() - time_start_step)

            # 存储转换数据 (显式拼接确保维度正确)
            time_start_store = time.time()
            transition = np.zeros(MEMORY_WIDTH)
            transition[0:7] = s
            transition[7:10] = [a_fc, a_bat, a_sc]
            transition[10] = r
            transition[11:18] = s_
            
            index = memory_counter[0] % MEMORY_CAPACITY
            shared_memory[index, :] = transition
            memory_counter[0] += 1
            episode_times['Data_Store'] += (time.time() - time_start_store)

            ep_r += r
            step_count += 1

            # 学习过程
            if memory_counter[0] > BATCH_SIZE and memory_counter[0] % LEARN_FREQUENCY == 0:
                time_start_learn = time.time()
                try:
                    l1 = FC_Agent.learn(0, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE) or 0.0
                    l2 = Bat_Agent.learn(1, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE) or 0.0
                    l3 = SC_Agent.learn(2, N_STATES, GAMMA, TARGET_REPLACE_ITER, BATCH_SIZE) or 0.0
                    step_loss.append((l1 + l2 + l3) / 3.0)
                except Exception as e:
                    print(f"\n⚠️ 学习过程异常 (Episode {i_episode}): {e}")
                    step_loss.append(0.0)
                episode_times['DQN_Learn'] += (time.time() - time_start_learn)

            if done:
                # 记录训练指标
                avg_loss = np.mean(step_loss) if step_loss else 0.0
                writer.add_scalar("Reward/Episode", ep_r, i_episode)
                writer.add_scalar("Loss/Average", avg_loss, i_episode)
                
                loss_records.append(avg_loss)
                using_time_total = time.time() - start_time_total
                current_lr = FC_Agent.optimizer.param_groups[0]["lr"]
                
                # 更新进度条
                episode_pbar.set_postfix({
                    'Rew': f'{ep_r:.2f}',
                    'LR': f'{current_lr:.2e}',
                    'Mem': f'{min(memory_counter[0], MEMORY_CAPACITY)}',
                    'Loss': f'{avg_loss:.4f}',
                    'Time': f'{using_time_total:.2f}s'
                })

                # 打印耗时分解（每500回合）
                if i_episode < 2 or (i_episode + 1) % 500 == 0:
                    print_time_breakdown(i_episode + 1, episode_times)
                break

            s = s_

        x_episodes.append(i_episode)
        y_rewards.append(ep_r)

        # 模型保存与早停逻辑（核心修改：保存包含RNN的完整模型）
        if ep_r > reward_max + REWARD_THRESHOLD:
            reward_max = ep_r
            reward_not_improve_episodes = 0
            # 保存包含RNN的完整最优模型
            torch.save(FC_Agent.eval_net.state_dict(), os.path.join(base_path, f"{best_model_base_name}_FC.pth"))
            torch.save(Bat_Agent.eval_net.state_dict(), os.path.join(base_path, f"{best_model_base_name}_BAT.pth"))
            torch.save(SC_Agent.eval_net.state_dict(), os.path.join(base_path, f"{best_model_base_name}_SC.pth"))
            # 额外保存独立的RNN模型（可选）
            torch.save(rnn_model.state_dict(), os.path.join(base_path, f"{best_model_base_name}_RNN.pth"))
            print(f"\n--- New Max Reward: {reward_max:.2f} ---")
            print(f"--- 已保存包含RNN的完整Joint模型到: {base_path} ---")
        else:
            reward_not_improve_episodes += 1

        # 学习率调度
        for agent in all_agents:
            agent.scheduler.step(ep_r)

        # 早停检查
        if reward_not_improve_episodes >= EARLY_STOP_PATIENCE:
            print(f"\n--- Early Stopping Triggered! (No improvement for {EARLY_STOP_PATIENCE} episodes) ---")
            training_done = True

    # 7. 最终处理（对齐train.py）
    final_episode = i_episode + 1 if not training_done else i_episode
    final_model_name = os.path.join(base_path, f"{best_model_base_name}_FINAL")
    
    # 保存包含RNN的最终完整模型
    torch.save(FC_Agent.eval_net.state_dict(), f"{final_model_name}_FC.pth")
    torch.save(Bat_Agent.eval_net.state_dict(), f"{final_model_name}_BAT.pth")
    torch.save(SC_Agent.eval_net.state_dict(), f"{final_model_name}_SC.pth")
    torch.save(rnn_model.state_dict(), f"{final_model_name}_RNN.pth")
    print(f"\nFinal models saved (包含RNN): {final_model_name}")

    # 整理训练最终指标
    final_metrics = {
        "max_reward": round(reward_max, 4),
        "final_reward": round(y_rewards[-1], 4) if y_rewards else 0,
        "average_reward": round(np.mean(y_rewards[POOL_SIZE:]) if len(y_rewards) > POOL_SIZE else 0, 4),
        "total_episodes_completed": final_episode,
        "early_stopped": training_done,
        "final_learning_rate": round(FC_Agent.optimizer.param_groups[0]["lr"], 6),
        "reward_not_improve_episodes": reward_not_improve_episodes,
        "best_model_reward": round(reward_max, 4),
        "excluded_episodes": POOL_SIZE
    }

    # 保存超参数
    save_hyperparameters(base_path, final_metrics)

    # 保存训练记录到CSV
    csv_path = os.path.join(base_path, "training_records.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("episode,reward,loss\n")
        for ep, r, l in zip(x_episodes, y_rewards, loss_records):
            f.write(f"{ep},{r:.4f},{l:.4f}\n")
    print(f"✅ 训练记录（含loss）已保存到CSV: {csv_path}")

    # 可视化与保存训练曲线
    writer.flush()
    writer.close()
    
    plt.figure(figsize=(12, 6))
    x_filtered = x_episodes[POOL_SIZE:]
    y_filtered = y_rewards[POOL_SIZE:]
    plt.plot(x_filtered, y_filtered, label='Episode Reward', color='#3570a8', linewidth=1.2)
    plt.axhline(y=reward_max, color='#c84343', linestyle='--', label=f'Best Reward: {reward_max:.2f}')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.title(f'JointNet Training Curve (Ep={final_episode}, Exclude First {POOL_SIZE} Episodes)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(base_path, f"{best_model_base_name}_train_curve.svg"), dpi=1200, bbox_inches='tight')
    plt.savefig(os.path.join(base_path, f"{best_model_base_name}_train_curve.png"), dpi=300, bbox_inches='tight')
    
    if REAL_TIME_DRAW:
        plt.ioff()
        plt.show()
    else:
        plt.close()

    print(f"\n🎉 JointNet训练完成！所有文件已保存到: {base_path}")
    print(f"\n📋 最优模型文件名前缀：{best_model_base_name}")
    print(f"📋 模型包含完整的RNN+MARL参数，后续可直接用--resume-training加载")

    # 8. 自动执行测试脚本（对齐train.py，修改RNN路径）
    test_script_path = os.path.join(project_root, "Scripts", "Chapter4", "test_Joint.py")
    if os.path.exists(test_script_path):
        test_cmd = [
            str(sys.executable),
            str(test_script_path),
            "--net-date", str(execute_date),
            "--train-id", str(train_id),
            "--model-prefix", str(best_model_base_name),
            "--rnn-path", os.path.join(base_path, f"{best_model_base_name}_RNN.pth")  # 使用训练后的RNN
        ]
        print("\n🚀 开始执行JointNet测试脚本...")
        print(" ".join(test_cmd))
        subprocess.run(test_cmd, check=True)
    else:
        print(f"\n⚠️ 测试脚本未找到: {test_script_path}，跳过自动测试")

    print(f"\n🎉 所有流程完成！文件保存路径: {base_path}")