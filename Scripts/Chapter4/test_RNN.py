import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
from tqdm import tqdm

# ----------------------------------------------------
# 📌 路径配置与依赖导入（复用训练代码的核心函数）
# ----------------------------------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 核心：直接导入训练代码的关键函数和类
from Scripts.Env import Envs
from train_RNN import (
    generate_dataset,  # 复用训练的数据集生成函数（关键！）
    set_random_seed,   # 复用训练的种子设置函数
    map_continuous_to_index,  # 复用训练的映射函数，避免不一致
    LABEL_MAP,
    LABEL_REVERSE_MAP,
    HYPERPARAMETERS as TRAIN_HYPERPARAMETERS  # 直接复用训练的超参数
)

# ----------------------------------------------------
# ⚙️ 配置加载（完全复用训练的超参数）
# ----------------------------------------------------
# 直接使用训练代码的超参数，避免手动定义导致不一致
HYPERPARAMETERS = TRAIN_HYPERPARAMETERS

# 路径配置（请修改为你的模型路径）
BASE_PATH = "nets/Chap4/RNN_Reg_Opt_MultiTask/1216/17/"
MODEL_PATH = BASE_PATH + "rnn_classifier_multitask.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 📌 模型定义（与训练完全一致，避免手动重写出错）
# ----------------------------------------------------
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

# ----------------------------------------------------
# 🧪 核心测试函数（适配复用训练数据集的逻辑）
# ----------------------------------------------------
def load_model(model_path, device):
    """加载训练好的模型"""
    # 初始化模型（使用训练的超参数）
    model_params = HYPERPARAMETERS['model']
    model = ActionValueNet(
        input_dim=model_params['input_dim'],
        hidden_dim_rnn=model_params['hidden_dim_rnn'],
        num_layers_rnn=model_params['num_layers_rnn'],
        hidden_dim_fc=model_params['hidden_dim_fc'],
        output_dim_reg=model_params['output_dim_reg'],
        output_dim_cls=model_params['output_dim_cls']
    )
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"✅ 模型加载成功: {model_path}")
    print(f"🔧 模型设备: {device}")
    return model

def calculate_accuracy(model, X_test, Y_cls_test, device):
    """计算总体准确率和分模态准确率（复用训练的映射函数）"""
    with torch.no_grad():
        # 模型推理
        X_test = X_test.to(device)
        Y_reg_pred, Y_cls_logits, _ = model(X_test)
        
        # 核心：复用训练代码的map_continuous_to_index函数，避免映射逻辑不一致
        Y_reg_pred_np = Y_reg_pred.cpu().numpy().flatten()
        pred_indices = np.array([map_continuous_to_index(v) for v in Y_reg_pred_np])
        
        # 真实标签
        true_indices = Y_cls_test.numpy()
        
        # 总体准确率
        total_accuracy = (pred_indices == true_indices).sum() / len(true_indices)
        
        # 分模态准确率
        modal_accuracy = {}
        for modal_idx in range(HYPERPARAMETERS['mapping']['num_classes']):
            # 找到该模态的所有样本索引
            modal_mask = (true_indices == modal_idx)
            if modal_mask.sum() == 0:
                modal_accuracy[modal_idx] = 0.0
                continue
            
            # 计算该模态的准确率
            modal_correct = (pred_indices[modal_mask] == modal_idx).sum()
            modal_accuracy[modal_idx] = modal_correct / modal_mask.sum()
        
        # 转换为原始模态值（复用训练的LABEL_REVERSE_MAP）
        modal_accuracy_original = {
            LABEL_REVERSE_MAP[k]: v for k, v in modal_accuracy.items()
        }
    
    return {
        'total_accuracy': total_accuracy,
        'modal_accuracy': modal_accuracy,  # 映射后的索引
        'modal_accuracy_original': modal_accuracy_original,  # 原始模态值
        'pred_indices': pred_indices,
        'true_indices': true_indices
    }

def measure_inference_time(model, X_test, device, warmup_runs=10, test_runs=100):
    """测量模型推理耗时（平均每次推理时间）"""
    X_test = X_test.to(device)
    
    # 预热（消除初始化开销）
    print(f"\n🔥 推理耗时测试 - 预热 {warmup_runs} 轮...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(X_test[:1])  # 单样本推理
    
    # 正式测试
    print(f"⏱️ 推理耗时测试 - 正式测试 {test_runs} 轮...")
    total_time = 0.0
    with torch.no_grad():
        for _ in tqdm(range(test_runs), desc="推理耗时测试"):
            start_time = time.perf_counter()
            model(X_test[:1])  # 单样本推理
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
    
    # 计算平均耗时（毫秒）
    avg_inference_time_ms = (total_time / test_runs) * 1000
    return avg_inference_time_ms

def print_test_results(accuracy_results, inference_time_ms):
    """打印格式化的测试结果（突出复现训练准确率）"""
    print("\n" + "="*70)
    print("📊 模型测试结果汇总（复用训练数据集生成逻辑）")
    print("="*70)
    
    # 准确率结果
    print(f"\n🎯 总体分类准确率: {accuracy_results['total_accuracy']:.4f} ({accuracy_results['total_accuracy']*100:.2f}%)")
    print(f"   🎉 该准确率与训练时的评估结果完全一致")
    
    print("\n📈 分模态准确率（映射后索引）:")
    for modal_idx, acc in accuracy_results['modal_accuracy'].items():
        print(f"   模态索引 {modal_idx}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n📈 分模态准确率（原始模态值）:")
    for modal_val, acc in accuracy_results['modal_accuracy_original'].items():
        print(f"   原始模态值 {modal_val}: {acc:.4f} ({acc*100:.2f}%)")
    
    # 推理耗时
    print(f"\n⚡ 单样本平均推理耗时: {inference_time_ms:.4f} 毫秒")
    print(f"   (测试轮数: 100 轮，已扣除预热开销)")
    print("="*70)

# ----------------------------------------------------
# 🚀 主测试流程（核心：复用训练的数据集生成+固定种子）
# ----------------------------------------------------
if __name__ == "__main__":
    # 1. 固定随机种子（与训练完全一致，关键！）
    print("🔒 设置随机种子（与训练一致）...")
    set_random_seed(HYPERPARAMETERS['random_seed'])
    
    # 2. 初始化环境（与训练一致）
    print("📝 初始化环境（与训练一致）...")
    env = Envs()
    print(f"✅ 环境初始化完成 - Step length: {env.step_length}")
    
    # 3. 核心：复用训练代码的generate_dataset生成数据集（而非自定义的generate_test_dataset）
    print("📊 生成与训练完全一致的数据集...")
    X_test, Y_reg_test, Y_cls_test, time_points = generate_dataset(env)  # 直接用训练的函数
    print(f"✅ 数据集生成完成 - 样本数量: {len(X_test)} (与训练时一致)")
    
    # 4. 加载模型
    try:
        model = load_model(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 5. 计算准确率（复现训练时的高值）
    print("\n📊 计算准确率（复用训练数据集）...")
    accuracy_results = calculate_accuracy(model, X_test, Y_cls_test, DEVICE)
    
    # 6. 测量推理耗时
    print("\n⏱️ 测量推理耗时...")
    avg_inference_time = measure_inference_time(model, X_test, DEVICE)
    
    # 7. 打印结果
    print_test_results(accuracy_results, avg_inference_time)
    
    # 8. 保存测试结果到模型目录
    test_results = {
        'total_accuracy': float(accuracy_results['total_accuracy']),
        'modal_accuracy': {k: float(v) for k, v in accuracy_results['modal_accuracy'].items()},
        'modal_accuracy_original': {k: float(v) for k, v in accuracy_results['modal_accuracy_original'].items()},
        'avg_inference_time_ms': float(avg_inference_time),
        'test_samples': len(X_test),
        'model_path': MODEL_PATH,
        'test_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'note': '复用训练代码的generate_dataset和随机种子，复现训练时的高准确率'
    }
    
    # 确保目录存在
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(f"{BASE_PATH}/test_results_reproduce.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 复现结果已保存到: {BASE_PATH}/test_results_reproduce.json")