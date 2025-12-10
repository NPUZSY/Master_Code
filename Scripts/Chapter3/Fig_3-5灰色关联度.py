import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # Base font size

def grey_relational_grade(data, ref_col=0, rho=0.5):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    ref = norm_data[:, ref_col].reshape(-1, 1)
    diff = np.abs(norm_data - ref)

    min_diff = diff.min()
    max_diff = diff.max()
    greys = (min_diff + rho * max_diff) / (diff + rho * max_diff)
    grades = greys.mean(axis=0)

    return grades, greys


# ===========================================
# 实验数据
# ===========================================

hyperparams = pd.DataFrame({
    "LR": [0.001, 0.005, 0.002],
    "Gamma": [0.90, 0.95, 0.98],
    "Tau": [0.40, 0.60, 0.80],
    "M": [10, 100, 1000]
})

experiment_output = [26.4167, 28.5782, 30.2521]

data_matrix = np.column_stack([experiment_output, hyperparams.values])
grades, details = grey_relational_grade(data_matrix, ref_col=0)

result = pd.DataFrame({
    "Factor": ["Experiment Output", "LR", "Gamma", "Tau", "M"],
    "Grey Relational Grade": grades
})

print("\n===== 灰色关联度结果 =====")
print(result)


# ===========================================
# 绘制灰色关联度柱状图（改进配色 + 数值标注）
# ===========================================

plt.figure(figsize=(8, 5), dpi=150)

# 使用更学术的配色方案
colors = plt.cm.Set2(np.linspace(0, 1, 4))

bars = plt.bar(
    result["Factor"][1:],
    result["Grey Relational Grade"][1:],
    width=0.55,
    color=colors
)

# 添加柱顶端数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{height:.3f}",
        ha='center', va='bottom',
        fontsize=10
    )

plt.ylabel("Grey Relational Grade", fontsize=12)
plt.xlabel("Hyperparameters", fontsize=12)
plt.title("Grey Relational Degree of Hyperparameters", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()


# ===========================================
# 保存图片 (PNG + SVG)
# ===========================================

save_dir = "../../Figures"
os.makedirs(save_dir, exist_ok=True)

png_path = os.path.join(save_dir, "Fig3-5-GRA_hyperparameters.png")
svg_path = os.path.join(save_dir, "Fig3-5-GRA_hyperparameters.svg")

plt.savefig(png_path, dpi=1200, format="png")
plt.savefig(svg_path, format="svg")

print(f"\n图片已保存：\n{png_path}\n{svg_path}")

plt.show()
