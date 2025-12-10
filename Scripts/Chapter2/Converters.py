import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ===================== 基础参数定义 =====================
# General parameters (adjust for actual hardware)
V_IN_NOMINAL = 48  # Nominal input voltage (V)
R_DS_ON = 0.01  # MOSFET on-resistance (Ω)
F_SW = 100e3  # Switching frequency (Hz)
Q_G = 100e-9  # MOSFET gate charge (C)
I_Q = 0.05  # Quiescent current (A)
V_D = 0.7  # Diode forward voltage drop (V)
LOSS_SW_COEFF = 1e-6  # Switching loss coefficient (empirical value)


# ===================== Efficiency Calculation Functions =====================
def buck_efficiency(P_in, I_in):
    """
    Calculate Buck converter efficiency
    :param P_in: Input power (W), scalar/array
    :param I_in: Input current (A), scalar/array
    :return: Efficiency (0-1)
    """
    # Avoid division by zero
    P_in = np.maximum(P_in, 0.1)
    I_in = np.maximum(I_in, 0.001)

    # Input voltage (derived from P_in = V_in * I_in)
    V_in = P_in / I_in

    # Buck converter duty cycle (fixed output voltage: 12V)
    V_out = 12
    D = V_out / V_in  # Ideal duty cycle

    # Estimate output current (initial efficiency assumption)
    I_out_est = P_in * 0.95 / V_out

    # Loss calculations
    loss_conduction = I_out_est ** 2 * R_DS_ON * D  # MOSFET conduction loss
    loss_switch = V_in * I_in * F_SW * Q_G * LOSS_SW_COEFF  # Switching loss
    loss_diode = I_out_est * V_D * (1 - D)  # Freewheeling diode loss
    loss_quiescent = V_in * I_Q  # Quiescent loss
    total_loss = loss_conduction + loss_switch + loss_diode + loss_quiescent

    # Output power and efficiency
    P_out = P_in - total_loss
    efficiency = np.clip(P_out / P_in, 0.6, 0.98)  # Efficiency range: 60%-98%
    return efficiency


def buck_boost_efficiency(P_in, I_in):
    """
    Calculate Buck-Boost converter efficiency
    :param P_in: Input power (W), scalar/array
    :param I_in: Input current (A), scalar/array
    :return: Efficiency (0-1)
    """
    # Avoid division by zero
    P_in = np.maximum(P_in, 0.1)
    I_in = np.maximum(I_in, 0.001)

    # Input voltage
    V_in = P_in / I_in

    # Buck-Boost output voltage (fixed: 24V)
    V_out = 24
    D = V_out / (V_in + V_out)  # Ideal duty cycle

    # Estimate output current (initial efficiency assumption)
    I_out_est = P_in * 0.9 / V_out  # Lower initial efficiency for Buck-Boost

    # Loss calculations (higher losses for Buck-Boost)
    loss_conduction = (I_in ** 2 * R_DS_ON * D) + (I_out_est ** 2 * R_DS_ON * (1 - D))
    loss_switch = V_in * I_in * F_SW * Q_G * LOSS_SW_COEFF * 1.2  # Higher switching loss
    loss_diode = I_out_est * V_D * D  # Diode loss
    loss_quiescent = V_in * I_Q * 1.1  # Higher quiescent loss
    total_loss = loss_conduction + loss_switch + loss_diode + loss_quiescent

    # Output power and efficiency
    P_out = P_in - total_loss
    efficiency = np.clip(P_out / P_in, 0.55, 0.95)  # Efficiency range: 55%-95%
    return efficiency


# ===================== Data Generation =====================
# Power range: 0-5000W, Current range: 0.1-100A (covers 5000W/50V=100A)
P_in_range = np.linspace(0, 5000, 100)  # Input power
I_in_range = np.linspace(0.1, 100, 100)  # Input current

# Create mesh grid
P_mesh, I_mesh = np.meshgrid(P_in_range, I_in_range)

# Calculate efficiency
eff_buck = buck_efficiency(P_mesh, I_mesh)
eff_buck_boost = buck_boost_efficiency(P_mesh, I_mesh)

# ===================== 3D Plotting (English + Times New Roman) =====================
# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # Base font size

# Create figure
fig = plt.figure(figsize=(16, 7))

# 1. Buck Converter Efficiency Plot
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(P_mesh, I_mesh, eff_buck,
                         cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_title('Buck Converter Efficiency Distribution\n(0-5000W)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Input Power (W)', fontsize=10)
ax1.set_ylabel('Input Current (A)', fontsize=10)
ax1.set_zlabel('Efficiency', fontsize=10)
ax1.set_zlim(0.6, 0.98)
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Efficiency')
cbar1.ax.set_ylabel('Efficiency', fontsize=10)

# 2. Buck-Boost Converter Efficiency Plot
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(P_mesh, I_mesh, eff_buck_boost,
                         cmap='plasma', alpha=0.8, edgecolor='none')
ax2.set_title('Buck-Boost Converter Efficiency Distribution\n(0-5000W)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Input Power (W)', fontsize=10)
ax2.set_ylabel('Input Current (A)', fontsize=10)
ax2.set_zlabel('Efficiency', fontsize=10)
ax2.set_zlim(0.55, 0.95)
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Efficiency')
cbar2.ax.set_ylabel('Efficiency', fontsize=10)

# Adjust layout
plt.tight_layout()

# ===================== Save Figures =====================
# Create directory if not exists
save_dir = "../../Figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define file name
file_name = "Fig2-9 Converters_Efficiency"

# Save as SVG (vector format)
svg_path = os.path.join(save_dir, f"{file_name}.svg")
plt.savefig(svg_path, format='svg', bbox_inches='tight')

# Save as PNG (1200 DPI, high resolution)
png_path = os.path.join(save_dir, f"{file_name}.png")
plt.savefig(png_path, format='png', dpi=1200, bbox_inches='tight')

# Show plot
plt.show()

print(f"Figures saved to:\n- SVG: {svg_path}\n- PNG: {png_path}")