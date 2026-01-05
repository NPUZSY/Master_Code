# 📘Meta-RL for CDV Energy Management
Reinforcement Learning–based Energy Management Strategy for Composite-Drive Vehicles

本仓库包含硕士论文《基于元强化学习的复合动力车辆能量管理策略研究》中全部相关代码，包括：

- 自定义复合动力系统能量管理环境（Fuel Cell + Battery + SuperCap）
- 多智能体强化学习（MARL）能量分配策略
- 元强化学习（Meta-RL）任务自适应训练框架
- 训练脚本、测试脚本、实验工具库
- 全部论文图表自动绘制脚本（SVG + 高清 PNG）

本仓库旨在提供一个可复现的 CDV 智能能量管理研究平台。

------------------------------------------------------------

## 📦Repository Structure
```commandline
Meta-RL-CDV/
├── Scripts/            # 强化学习环境与算法实现
│   ├── Env.py          # 自定义三能源系统环境（FC + Battery + SuperCap）
│   ├── Chapter2/       # 第二章：基础理论与系统建模
│   ├── Chapter3/       # 第三章：多智能体强化学习（MARL）
│   ├── Chapter4/       # 第四章：元强化学习（Meta-RL）
│   ├── Chapter5/       # 第五章：快速适应测试与性能分析
│   ├── utils/          # 工具函数库
│   ├── train.py        # 基础训练脚本
│   └── test.py         # 基础测试脚本
├── nets/               # 训练好的模型权重（按实验/章节管理）
├── logs/               # 训练日志与实验记录
├── Data/               # 实验数据文件
├── Figures/            # 论文图表输出（SVG + 高分辨率 PNG）
├── environment.yml     # Conda 环境配置文件（用于一键复现）
└── Readme.md           # 项目说明文档
```

------------------------------------------------------------

## 🚗System Description

### 1. 复合动力系统

CDV（Composite Drive Vehicle）由三种能源构成：

- **Fuel Cell（燃料电池）**：提供持续稳定的功率输出
- **Battery（锂电池）**：提供中等功率和能量密度
- **Supercapacitor（超级电容）**：提供快速响应的峰值功率

### 2. 能量管理目标

- 满足功率需求与系统安全约束
- 降低氢耗，提高能源利用效率
- 保护电池寿命，减少电池循环次数
- 提高对复杂多变任务的自适应能力
- 快速适应新环境和新工况

### 3. 关键特性

- **多智能体协同**：三种能源各有独立智能体，协同优化能量分配
- **元学习能力**：通过元强化学习快速适应新场景
- **双重触发机制**：基于KL散度和性能指标的在线更新触发
- **详细计时统计**：全面的性能计时，便于分析和优化

### 4. 应用场景

- **巡航场景（Cruise）**：长航时稳定运行
- **侦察场景（Recon）**：跨域快速机动
- **救援场景（Rescue）**：高强度功率需求

## 🤖Algorithms Included

1. **Multi-Agent Reinforcement Learning (MARL)**
   - Independent Q-Learning (IQL)
   - 三智能体结构：FC-Agent / BAT-Agent / SC-Agent

2. **Meta-Reinforcement Learning (Meta-RL)**
   - 基于任务嵌入的自适应策略
   - 支持 Few-shot 快速微调

3. **Baseline 控制策略**
   - ECMS（等效消耗最小化策略）
   - Rule-based EMS（基于规则的能量管理策略）

4. **快速适应算法**
   - 环境分布差异度量（KL散度）
   - 滑动窗口采样与核密度估计
   - 双重触发机制：KL散度阈值 + 性能指标阈值
   - 在线更新流程：参数备份 → 局部优化 → 效果验证

------------------------------------------------------------

## 🧪How to Run

### 1. 环境配置
```bash
# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate Meta-RL-310
```

### 2. 训练与测试

#### 第三章：多智能体强化学习（MARL）
```bash
# 训练 MARL 模型
python Scripts/Chapter3/train_MARL.py

# 测试 MARL 模型
python Scripts/Chapter3/test.py
```

#### 第四章：元强化学习（Meta-RL）
```bash
# 训练 Meta-RL 模型
python Scripts/Chapter4/train_meta_policy.py
```

#### 第五章：快速适应测试与性能分析
```bash
# 运行快速适应测试
python Scripts/Chapter5/fast_adaptation.py

# 可选参数
python Scripts/Chapter5/fast_adaptation.py --scenario cruise --max-steps 1000 --episodes 5
```

### 3. 输出结果

运行后会自动生成：

- 三能源功率分配曲线
- SOC 演化曲线
- 系统温度曲线
- 超级电容能量流动统计
- 氢耗分解与优化分析
- 强化学习性能指标
- **详细的计时统计**：
  - 总测试耗时
  - 平均每回合耗时
  - 平均每步耗时
  - 智能体决策耗时
  - 模型更新耗时

### 4. 结果保存

- **模型权重**：保存在 `nets/` 目录下，按章节和实验命名
- **实验日志**：保存在 `logs/` 目录下
- **图表输出**：保存在 `Figures/` 目录下，包含 SVG 和高清 PNG 格式
- **测试结果**：以 JSON 格式保存，包含详细的性能数据和计时统计


------------------------------------------------------------

## 📊Example Outputs

### 1. 性能可视化

所有论文图表均自动生成，包括：

- **功率分配曲线**：三种能源的实时功率输出
- **SOC 演化曲线**：电池和超级电容的荷电状态变化
- **温度曲线**：系统温度动态变化
- **氢耗分析**：氢耗分解与优化效果
- **能量流动图**：能量在各组件间的流动关系

### 2. 计时统计示例

运行第五章快速适应测试后，会输出详细的计时统计：

```
✅ 场景 cruise 测试完成
   总奖励: 1234.56
   平均奖励: 12.3456
   总步数: 1000

   ⏱️  耗时统计:
   总测试耗时: 12.3456秒
   平均每回合耗时: 2.4692秒
   平均每步耗时: 0.012345秒
   最大单步耗时: 0.1234秒
   最小单步耗时: 0.001234秒
   平均决策耗时: 0.005678秒
   最大决策耗时: 0.0567秒
   最小决策耗时: 0.001234秒
   总更新次数: 5
   总更新耗时: 1.2345秒
   平均每次更新耗时: 0.2469秒
```

### 3. 图表格式

- **SVG 矢量图**：用于论文排版和编辑
- **PNG 高清图**：DPI 1200，用于演示和报告

### 4. 结果文件

- **JSON 格式**：包含所有测试数据、性能指标和计时统计
- **numpy 数组**：原始实验数据，用于进一步分析

------------------------------------------------------------

## 🔍Citation

如在研究中使用本仓库，请引用：

Z. S. Yuan, “Meta-Reinforcement Learning for Composite-Drive Vehicle Energy Management,”  
Master Thesis, Northwestern Polytechnical University, 2025.


------------------------------------------------------------

## 📝License

本仓库仅限科研与学术用途。商业用途请联系作者。

------------------------------------------------------------

## 🙌Acknowledgements

本项目部分参考：

- PyTorch
- Gymnasium
- Stable-Baselines3
- Farama Foundation
- 相关开源研究工作

------------------------------------------------------------