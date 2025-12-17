import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import datetime
import os
import sys
import matplotlib.pyplot as plt
import json 
import random 
from tqdm import tqdm # 🚨 新增：导入 tqdm 库

# ----------------------------------------------------
# ⚙️ 超参数配置区域 (Hyperparameter Configuration)
# ----------------------------------------------------
# 📌 针对回归任务的优化配置
HYPERPARAMETERS = {
    # 模型架构参数 - 🚨 结构修改：GRU层数增加到2, GRU维度增加到256
    'model': {
        'input_dim': 7,
        'hidden_dim_rnn': 256,   # 增加到 256
        'num_layers_rnn': 2,   # 新增：GRU 层数
        'hidden_dim_fc': 64,
        'output_dim_reg': 1,   # 回归输出维度
        'output_dim_cls': 4,   # 分类输出维度 (4类)
    },
    # 训练参数 - 多阶段配置
    'training': {
        'total_epochs': 10000,   # 总训练轮数
        'batch_size': 64,   
        # 多阶段学习率配置
        'lr_schedule': [
            {'epochs': 1000, 'lr': 5e-3}, 
            {'epochs':9000, 'lr': 1e-3} 
        ],
        'save_path_base': "nets/Chap4/RNN_Reg_Opt_MultiTask", # 区分路径
        # 🚨 新增：多任务损失权重
        'loss_weights': {
            'mae_weight': 1.0,   # 回归损失权重
            'ce_weight': 0.5    # 分类损失权重 (LCE / LTotal ≈ 1/3)
        }
    },
    # 标签/映射参数
    'mapping': {
        'ranges': [0.0, 0.25, 0.5, 0.75, 1.0], 
        'num_classes': 4,
    },
    # 🚨 新增：随机种子配置
    'random_seed': 42 
}
# ----------------------------------------------------


# ----------------------------------------------------
# 📌 显式路径导入 (保持不变)
# ----------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
print(f"Project root manually added to sys.path: {project_root}")

from Scripts.Chapter3.MARL_Engine import font_get, get_max_folder_name
from Scripts.Env import Envs


# --- ActionValueNet 定义 (保持不变) ---
class ActionValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_rnn, num_layers_rnn, hidden_dim_fc, output_dim_reg, output_dim_cls):
        super(ActionValueNet, self).__init__()
        
        self.rnn = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim_rnn, 
            num_layers=num_layers_rnn,  
            batch_first=True
        )
        
        self.fc_rnn_to_64 = nn.Linear(hidden_dim_rnn, hidden_dim_fc)
        self.reg_head = nn.Linear(hidden_dim_fc, output_dim_reg)
        self.cls_head = nn.Linear(hidden_dim_fc, output_dim_cls)
        self.requires_grad_fc_64_1_only = False
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        out_rnn, _ = self.rnn(x) 
        feature_rnn = out_rnn.squeeze(1) 
        feature_64 = F.relu(self.fc_rnn_to_64(feature_rnn))

        a_raw_reg = self.reg_head(feature_64) 
        a_out_reg = torch.sigmoid(a_raw_reg)
        
        a_out_cls_logits = self.cls_head(feature_64)
        
        return a_out_reg, a_out_cls_logits, feature_64
        
    def set_trainable_layers(self, trainable=True):
        for param in self.parameters(): 
            param.requires_grad = trainable

font_get()

# ----------------------------------------------------
# 数据集生成和标签映射 (保持不变)
# ----------------------------------------------------
LABEL_MAP = {1: 0, 0: 1, 2: 2, 3: 3} 
LABEL_REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}
TIME_RANGES = [(0, 150), (150, 200), (200, 400), (400, 450), (450, 600), (600, 650), (650, 800)]
TIME_LABELS = [1, 0, 2, 0, 3, 0, 1]

def index_to_target(index):
    num_classes = HYPERPARAMETERS['mapping']['num_classes']
    if num_classes <= 1: return 0.0
    return index / (num_classes - 1)

def get_label_by_time(time_stamp):
    for (start, end), label_val in zip(TIME_RANGES, TIME_LABELS):
        if start <= time_stamp < end:
            return LABEL_MAP.get(label_val)
    return LABEL_MAP.get(TIME_LABELS[-1]) 

def generate_dataset(env):
    observations = []; target_reg_labels = []; target_cls_labels = []; time_points = []
    N_FC, N_BAT, N_SC = env.N_FC_ACTIONS, env.N_BAT_ACTIONS, env.N_SC_ACTIONS
    obs = env.reset(); done = False
    
    while not done:
        time_sec = env.time_stamp * env.dt
        time_points.append(time_sec)
        observations.append(obs[:7])
        
        label_idx = get_label_by_time(time_sec)
        
        target_value = index_to_target(label_idx) 
        target_reg_labels.append(target_value)
        target_cls_labels.append(label_idx)
        
        a_fc = np.random.randint(0, N_FC); a_bat = np.random.randint(0, N_BAT); a_sc = np.random.randint(0, N_SC)
        action_list = [a_fc, a_bat, a_sc]
        obs, _, done, _ = env.step(action_list)
        
        if env.time_stamp >= env.step_length - 1:
            break
            
    X = torch.tensor(np.array(observations), dtype=torch.float32)
    Y_reg = torch.tensor(np.array(target_reg_labels), dtype=torch.float32).unsqueeze(1) 
    Y_cls = torch.tensor(np.array(target_cls_labels), dtype=torch.long)
    
    return X, Y_reg, Y_cls, np.array(time_points)

def map_continuous_to_index(continuous_value):
    ranges = HYPERPARAMETERS['mapping']['ranges']
    if continuous_value < ranges[1]: return 0
    elif continuous_value < ranges[2]: return 1
    elif continuous_value < ranges[3]: return 2
    else: return 3
    
def target_to_index(target_value):
    num_classes = HYPERPARAMETERS['mapping']['num_classes']
    if num_classes <= 1: return 0
    return int(np.round(target_value * (num_classes - 1)))


# ----------------------------------------------------
# 绘图函数 (保持不变)
# ----------------------------------------------------
def plot_loss(loss_history, accuracy_history, save_dir, lr_schedule):
    fig_path = os.path.join(save_dir, "RNN_Loss_Accuracy_MultiTask.png") 
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- 绘制 Loss (左 Y 轴) ---
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (Weighted MAE + CE)', color='blue')
    ax1.plot(loss_history, label='Total Training Loss', color='blue', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- 绘制 Accuracy (右 Y 轴) ---
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Classification Accuracy', color='red')
    ax2.plot(accuracy_history, label='Classification Accuracy', color='red', linestyle='-', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.05) 
    
    # 标注学习率阶段 
    current_epoch = 0
    colors = ['red', 'green', 'purple', 'orange']
    font_size_large = 20 
    
    for i, phase in enumerate(lr_schedule):
        phase_epochs = phase['epochs']
        phase_lr = phase['lr']
        start_epoch = current_epoch
        end_epoch = current_epoch + phase_epochs
        
        if end_epoch > len(loss_history):
            end_epoch = len(loss_history)
        
        if i > 0 and start_epoch < len(loss_history):
            ax1.axvline(x=start_epoch, color='black', linestyle='--', alpha=0.5)
        
        if start_epoch < end_epoch:
            mid_epoch = (start_epoch + end_epoch) // 2
            
            ax2.text(mid_epoch, 
                     1.0, # 固定 Y 坐标在 1.0 (右轴的顶部)
                     f'LR={phase_lr}', 
                     ha='center', 
                     va='top', # 设置垂直对齐方式为 'top'，确保文字底部在 Y=1.0
                     color=colors[i % len(colors)], 
                     fontweight='bold',
                     fontsize=font_size_large) # 字号变大一倍
        
        current_epoch += phase_epochs
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.title('Training Loss and Classification Accuracy - Multi-Task Learning') 
    
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(fig_path, format='png', dpi=1200)
    plt.close()
    print(f"🖼️ Loss/Accuracy plot saved to: {fig_path}")

def plot_results(time_points, true_target_vals, predicted_continuous_vals, save_dir):
    fig_path = os.path.join(save_dir, "RNN_Result_MultiTask.png")
    
    predicted_indices = np.array([map_continuous_to_index(v) for v in predicted_continuous_vals.flatten()])
    true_indices = np.array([target_to_index(v) for v in true_target_vals.flatten()])
    predicted_vals = np.array([LABEL_REVERSE_MAP[idx] for idx in predicted_indices])
    true_vals = np.array([LABEL_REVERSE_MAP[idx] for idx in true_indices])
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_points, true_target_vals.flatten(), label='True Target Value (Normalized Index)', color='black', linestyle='--')
    plt.plot(time_points, predicted_continuous_vals.flatten(), label='Predicted Continuous Value', color='blue', alpha=0.7)
    plt.title('RNN Regression Output (0 to 1) - Multi-Task')
    plt.xlabel('Time (s)')
    plt.ylabel('Regression Value')
    plt.yticks(np.linspace(0, 1, 5))
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.step(time_points, true_vals, where='post', label='True Mode (Ground Truth)', color='black', linestyle='--')
    plt.step(time_points, predicted_vals, where='post', label='RNN Predicted Mode (Mapped)', color='red', alpha=0.7)
    plt.title('Operating Mode Classification Results (Mapped) - Multi-Task')
    plt.xlabel('Time (s)')
    plt.ylabel('Operating Mode Value')
    mode_values = sorted(list(LABEL_MAP.keys()))
    plt.yticks(mode_values)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(fig_path, format='png', dpi=1200)
    plt.close()
    print(f"🖼️ Results plot saved to: {fig_path}")

    accuracy = (predicted_indices == true_indices).sum() / len(true_indices)
    print(f"📊 Classification Accuracy (Mapped): {accuracy:.4f}")
    return accuracy


# ----------------------------------------------------
# 📌 全局随机种子设置函数 (保持不变)
# ----------------------------------------------------
def set_random_seed(seed):
    """设置所有使用的库的随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")


# ----------------------------------------------------
# 训练脚本主体 (🚨 核心修改：集成 tqdm)
# ----------------------------------------------------

def train_rnn_classifier(model, X, Y_reg, Y_cls, time_points, params):
    set_random_seed(params['random_seed']) 

    total_epochs = params['training']['total_epochs']
    batch_size = params['training']['batch_size']
    lr_schedule = params['training']['lr_schedule']
    mae_weight = params['training']['loss_weights']['mae_weight']
    ce_weight = params['training']['loss_weights']['ce_weight']
    
    # 1. 路径生成和保存... 
    now = datetime.datetime.now()
    mmdd = now.strftime("%m%d")
    base_save_path = os.path.join(project_root, params['training']['save_path_base'], mmdd)
    next_index = get_max_folder_name(base_save_path) 
    save_dir = os.path.join(base_save_path, str(next_index))
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "rnn_classifier_multitask.pth")
    config_save_path = os.path.join(save_dir, "config.json") 
    figures_save_dir = save_dir 
    print(f"📁 Training run index: {next_index}")
    
    with open(config_save_path, 'w') as f:
        json.dump(params, f, indent=4)

    loss_history = []
    accuracy_history = []
    
    # 2. 模型、损失函数和优化器 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion_reg = nn.L1Loss()
    criterion_cls = nn.CrossEntropyLoss()
    
    true_indices_np = Y_cls.numpy()
    num_samples = X.size(0)
    
    initial_lr = lr_schedule[0]['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) 

    # 3. 多阶段训练循环
    print(f"🚀 Starting multi-task training on {device}...")
    model.set_trainable_layers(trainable=True) 
    
    phase_index = 0
    phase_epochs = lr_schedule[phase_index]['epochs']
    current_lr = lr_schedule[phase_index]['lr']
    
    # 🚨 外部循环：使用 tqdm 包装 Epoch 循环
    epoch_loop = tqdm(range(total_epochs), desc="Training Progress")
    
    for epoch in epoch_loop:
        if epoch >= phase_epochs and phase_index < len(lr_schedule) - 1:
            phase_index += 1
            current_lr = lr_schedule[phase_index]['lr']
            phase_epochs += lr_schedule[phase_index]['epochs'] 
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # 当 LR 切换时，打印信息，tqdm 会自动换行
            print(f"\n🔄 Switching to Phase {phase_index+1} - LR updated to {current_lr}")
        
        model.train()
        permutation = torch.randperm(num_samples)
        total_loss_sum = 0
        
        # 内部循环：使用 tqdm 包装 Batch 循环
        num_batches = (num_samples + batch_size - 1) // batch_size
        batch_iterator = range(0, num_samples, batch_size)
        
        # 使用 tqdm 包装 range 以显示 Batch 进度
        for i in batch_iterator:
            indices = permutation[i:i + batch_size]
            batch_x = X[indices].to(device)
            batch_y_reg = Y_reg[indices].to(device)
            batch_y_cls = Y_cls[indices].to(device)
            
            predicted_y_reg, predicted_y_cls_logits, _ = model(batch_x)
            
            loss_reg = criterion_reg(predicted_y_reg, batch_y_reg)
            loss_cls = criterion_cls(predicted_y_cls_logits, batch_y_cls)
            loss = mae_weight * loss_reg + ce_weight * loss_cls
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_sum += loss.item() * batch_x.size(0)
            
        avg_loss = total_loss_sum / num_samples
        loss_history.append(avg_loss)

        # 准确率计算 (每个 Epoch 记录)
        model.eval()
        with torch.no_grad():
            predicted_y_reg_full, _, _ = model(X.to(device))
            predicted_continuous_vals = predicted_y_reg_full.cpu().numpy().flatten()
            predicted_indices_reg_map = np.array([map_continuous_to_index(v) for v in predicted_continuous_vals])
            accuracy = (predicted_indices_reg_map == true_indices_np).sum() / num_samples
            accuracy_history.append(accuracy) 
            
        model.train()
        
        # 🚨 更新 tqdm 描述信息，显示当前 Loss 和 Accuracy
        epoch_loop.set_postfix(
            Loss=f"{avg_loss:.4f}", 
            Acc=f"{accuracy:.4f}", 
            LR=f"{current_lr:.1e}"
        )
        
        # 移除原有的每100个epoch打印语句，因为 tqdm 已经提供了实时信息
        # if (epoch + 1) % 100 == 0 or epoch == total_epochs - 1: 
        #      print(f"Epoch {epoch + 1}/{total_epochs} | LR={current_lr:.6f} | Total Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")

    # 4. 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete. Model saved to {model_save_path}")

    # 5. 绘图 (保持不变)
    print("🎨 Generating plots...")
    plot_loss(loss_history, accuracy_history, figures_save_dir, lr_schedule) 
    
    model.eval()
    with torch.no_grad():
        predicted_y_reg_full, _, _ = model(X.to(device))
        predicted_continuous_vals = predicted_y_reg_full.cpu().numpy()
        true_target_vals = Y_reg.cpu().numpy()
    
    plot_results(time_points, true_target_vals, predicted_continuous_vals, figures_save_dir)


# ----------------------------------------------------
# 主执行区 (Main Execution - 保持不变)
# ----------------------------------------------------

if __name__ == "__main__":
    SEED = HYPERPARAMETERS['random_seed']
    set_random_seed(SEED)
    
    # 1. 实例化环境
    try:
        env = Envs()
        print(f"Environment initialized. Step length: {env.step_length} steps.")
    except Exception as e:
        print(f"Error initializing Env: {e}")
        exit()
        
    # 2. 生成数据集
    print("⏳ Generating dataset from environment...")
    X_train, Y_reg_train, Y_cls_train, time_points = generate_dataset(env)
    print(f"Dataset generated. X shape: {X_train.shape}, Y_reg shape: {Y_reg_train.shape}, Y_cls shape: {Y_cls_train.shape}")
    
    # 3. 实例化 RNN 模型 
    model_params = HYPERPARAMETERS['model']
    rnn_model = ActionValueNet(
        input_dim=model_params['input_dim'], 
        hidden_dim_rnn=model_params['hidden_dim_rnn'], 
        num_layers_rnn=model_params['num_layers_rnn'],
        hidden_dim_fc=model_params['hidden_dim_fc'], 
        output_dim_reg=model_params['output_dim_reg'],
        output_dim_cls=model_params['output_dim_cls']
    )
    
    # 4. 运行训练 
    train_rnn_classifier(rnn_model, X_train, Y_reg_train, Y_cls_train, time_points, params=HYPERPARAMETERS)