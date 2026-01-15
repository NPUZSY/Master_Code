#!/usr/bin/env python3
import json
import numpy as np

# 读取测试结果文件
file_path = '/home/siyu/Master_Code/nets/Chap5/fast_adaptation/0114_145301/fast_adaptation_result_cruise.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# 获取KL散度值
kl_values = data['kl_values']
print(f"KL散度值数量: {len(kl_values)}")

# 计算统计信息
print(f"KL散度最小值: {np.min(kl_values):.4f}")
print(f"KL散度最大值: {np.max(kl_values):.4f}")
print(f"KL散度平均值: {np.mean(kl_values):.4f}")
print(f"KL散度中位数: {np.median(kl_values):.4f}")

# 查看前50个值
print("\n前50个KL散度值:")
for i, kl in enumerate(kl_values[:50]):
    print(f"步骤 {i}: {kl:.4f}")

# 找出KL散度大于阈值的点
kl_threshold = 0.3  # 与超参数中的kl_threshold一致
high_kl_points = [(i, kl) for i, kl in enumerate(kl_values) if kl > kl_threshold]
print(f"\nKL散度大于阈值 {kl_threshold} 的点数量: {len(high_kl_points)}")
print("KL散度大于阈值的点:")
for i, kl in high_kl_points[:20]:  # 显示前20个
    print(f"步骤 {i}: {kl:.4f}")
