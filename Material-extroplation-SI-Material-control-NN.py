import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# =====================================================================
# 🌟 模块 1：底层物理方程与基准解析解 (原封不动保留你的智慧)
# =====================================================================
def solve_3layer_cylinder(R, E, nu, alpha, dT, outer_bc='fixed'):
    R1, R2, R3 = R
    E1, E2, E3 = E
    n1, n2, n3 = nu
    a1, a2, a3 = alpha
    
    V1, V2, V3 = R1**2, R2**2 - R1**2, R3**2 - R2**2
    K, F_vec = np.zeros((6, 6)), np.zeros(6)
    
    K[0,0], K[0,1], K[0,2] = 1.0, -1.0, 1.0/(R1**2)
    K[1,0], K[1,1], K[1,2], K[1,5] = (1+n1)*(1-2*n1)/E1, -(1+n2)*(1-2*n2)/E2, -(1+n2)/(E2*R1**2), -(n1-n2)
    F_vec[1] = (1+n2)*a2*dT - (1+n1)*a1*dT
    
    K[2,1], K[2,2], K[2,3], K[2,4] = 1.0, -1.0/(R2**2), -1.0, 1.0/(R2**2)
    K[3,1], K[3,2], K[3,3], K[3,4], K[3,5] = (1+n2)*(1-2*n2)/E2, (1+n2)/(E2*R2**2), -(1+n3)*(1-2*n3)/E3, -(1+n3)/(E3*R2**2), -(n2-n3)
    F_vec[3] = (1+n3)*a3*dT - (1+n2)*a2*dT
    
    if outer_bc == 'fixed':
        K[4,3], K[4,4], K[4,5] = (1+n3)*(1-2*n3)/E3, (1+n3)/(E3*R3**2), -n3
        F_vec[4] = -(1+n3)*a3*dT
        
    K[5,0], K[5,1], K[5,3], K[5,5] = 2*n1*V1, 2*n2*V2, 2*n3*V3, E1*V1 + E2*V2 + E3*V3
    F_vec[5] = (E1*V1*a1 + E2*V2*a2 + E3*V3*a3)*dT
    
    X = np.linalg.solve(K, F_vec)
    return {'A1':X[0], 'A2':X[1], 'B2':X[2], 'A3':X[3], 'B3':X[4], 'eps_0':X[5]}

def calculate_mises_stress(r, layer_idx, constants, E, nu, alpha, dT):
    eps_0 = constants['eps_0']
    if layer_idx == 1: A, B = constants['A1'], 0.0
    elif layer_idx == 2: A, B = constants['A2'], constants['B2']
    elif layer_idx == 3: A, B = constants['A3'], constants['B3']
    
    sigma_r = A - B/(r**2)
    sigma_theta = A + B/(r**2)
    sigma_z = E[layer_idx-1]*(eps_0 - alpha[layer_idx-1]*dT) + 2*nu[layer_idx-1]*A
    
    term1 = (sigma_r - sigma_theta)**2
    term2 = (sigma_theta - sigma_z)**2
    term3 = (sigma_z - sigma_r)**2
    return np.sqrt(0.5 * (term1 + term2 + term3))

def get_eshelby_3d_stress(a, b, nu, Sr, Sz):
    c = b
    term1 = (a / c) * np.sqrt((a / c)**2 - 1)
    term2 = np.arccosh(a / c)
    Ic = (2 * np.pi * a * c**2 / (a**2 - c**2)**1.5) * (term1 - term2)
    
    Ia = 4 * np.pi - 2 * Ic
    Iac = (Ic - Ia) / (3 * (a**2 - c**2))
    Iaa = (4 * np.pi / (3 * a**2)) - 2 * Iac
    Ibc = (np.pi / (3 * c**2)) - 0.25 * Iac
    Icc = 3 * Ibc
    
    Q = 3 / (8 * np.pi * (1 - nu))
    R = (1 - 2 * nu) / (8 * np.pi * (1 - nu))
    
    S11 = Q * a**2 * Iaa + R * Ia
    S33 = Q * c**2 * Icc + R * Ic
    S13 = Q * c**2 * Iac - R * Ia 
    S31 = Q * a**2 * Iac - R * Ic 
    S23 = Q * c**2 * Ibc - R * Ic 
    
    e11_A = Sz - 2 * nu * Sr
    e33_A = (1 - nu) * Sr - nu * Sz
    
    A11, A12 = 1 - S11, -2 * S13
    A21, A22 = -S31, 1 - S33 - S23
    
    det = A11 * A22 - A12 * A21
    e11_T = (e11_A * A22 - e33_A * A12) / det
    e33_T = (A11 * e33_A - A21 * e11_A) / det
    
    Sigma_z = (e11_T + nu * e33_T) / (1 - nu**2)
    Sigma_theta = (e33_T + nu * e11_T) / (1 - nu**2)
    
    return Sigma_z, Sigma_theta


# =====================================================================
# 🌟 模块 2：常数定义与 10 个极值样本提取
# =====================================================================
pi = 3.1415926536
L2=25; E2=70000; aa2=6e-7; v2=0.16; h2=0.5
E3=129000; aa3=3e-6; v3=0.28
T=125

print("加载并提取无量纲物理特征...")
materials_E = [410000.0, 280000.0, 220000.0, 130000.0, 70000.0]
materials_alpha = [4.5e-6, 7.5e-6, 12e-6, 16e-6, 18.0e-6]
materials_nu = [0.28, 0.31, 0.30, 0.31, 0.30]
mat_idx = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

# 原始几何输入
geom_raw = [
    [1.0, 20.0, 0.6], [1.8, 15.0, 0.5], # W 
    [1.0, 25.0, 0.4], [1.8, 25.0, 0.4], # IN1
    [1.0, 15.0, 0.6], [1.8, 25.0, 0.4], # IN3 
    [1.0, 15.0, 0.6], [1.8, 25.0, 0.4], # OUT3 
    [1.0, 25.0, 0.5], [1.8, 25.0, 0.4]  # CU 
]

true_fea_stress = torch.tensor([
    70.26367188, 74.31912231, 79.16014862, 101.9628983, 108.1549835, 
    154.6559753, 124.3113708, 192.584549, 115.8892365, 183.3897705
], dtype=torch.float32)

# 动态计算每个样本的四大无量纲描述符、mises 理论解、以及 s_target
X_features = []
mises_theory_list = []
s_targets_list = []

for i in range(10):
    E_t = materials_E[mat_idx[i]]
    a_t = materials_alpha[mat_idx[i]]
    nu_t = materials_nu[mat_idx[i]]
    
    current_h1 = geom_raw[i][0] 
    a_val = geom_raw[i][1] 
    b_val = geom_raw[i][2] 
    
    R_current = [current_h1, current_h1 + h2, 5.0]
    constants = solve_3layer_cylinder(
        R_current, [E_t, E2, E3], [nu_t, v2, v3], [a_t, aa2, aa3], T, outer_bc='fixed'
    )
    
    m_theory = calculate_mises_stress(
        current_h1 + h2, 3, constants, [E_t, E2, E3], [nu_t, v2, v3], [a_t, aa2, aa3], T
    )
    mises_theory_list.append(m_theory)
    
    Sr_actual = constants['A3']
    Sz_actual = E3 * (constants['eps_0'] - aa3 * T) + 2 * v3 * constants['A3']
    ratio_zr = Sz_actual / (Sr_actual + 1e-9)
    K_z, K_theta = get_eshelby_3d_stress(a_val, b_val, v3, 1.0, ratio_zr)
    
    h3 = 5.0 - current_h1 - h2
    k_val = (h3/current_h1 * E3/E_t + 1)**0.5
    n1_val = ((E2/(2*(1+v2)*E3*h2*h3))**0.5)*L2
    
    X_features.append([K_theta, K_z, k_val, n1_val])
    
    # 动态标度 s 计算
    s_target = E_t * abs(a_t - 3e-6) * T / (1 - nu_t)
    s_targets_list.append(s_target)

# 转为 Tensor
mises_theory = torch.tensor(mises_theory_list, dtype=torch.float32)
s_targets = torch.tensor(s_targets_list, dtype=torch.float32)

# =====================================================================
# 🌟 模块 3：特征归一化 (直接使用官方 Scaler 对接无量纲数据)
# =====================================================================
scaler_X = joblib.load("scaler_w_SI_geom_features.pkl")
X_scaled_np = scaler_X.transform(np.array(X_features)) # 100% 完美的无损映射
geom_inputs = torch.tensor(X_scaled_np, dtype=torch.float32)

# 材料特征归一化
E_raw = torch.tensor([materials_E[i] for i in mat_idx], dtype=torch.float32)
alpha_raw = torch.tensor([materials_alpha[i] for i in mat_idx], dtype=torch.float32)
E_norm = E_raw / 10000.0      
alpha_norm = alpha_raw * 1e6                
mat_inputs = torch.stack([E_norm, alpha_norm], dim=1)


# =====================================================================
# 🌟 模块 4：网络架构组装 (双流网络闭环)
# =====================================================================
# 1. 冻结的 Base-NN (纯几何特征提取器) 
class PreTrainedBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 完美复刻你的原始代码！(4->256->256->1)
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.model(x).squeeze()

base_nn = PreTrainedBaseNet()
# TODO: 请确保当前目录下有这个 W 的预训练权重文件
base_nn.load_state_dict(torch.load("W-SI-GEOM-RESIDUAL.pth"))
base_nn.eval()
for param in base_nn.parameters():
    param.requires_grad = False

# 2. 全新的 Micro-NN (材料本构推断器)
class MaterialMicroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Mish(),
            nn.Linear(16, 16), nn.Mish(),
            nn.Linear(16, 2)
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[4].weight)
        self.net[4].bias.data = torch.tensor([1.0, 0.0]) 

    def forward(self, x):
        out = self.net(x)
        return out[:, 0], out[:, 1]

micro_nn = MaterialMicroNet()

# =====================================================================
# 🌟 模块 5：端到端反向传播 (物理逼近极小化)
# =====================================================================
print("\n开始端到端物理标定训练...")
optimizer = optim.Adam(micro_nn.parameters(), lr=0.01)
# 当 Loss 连续 300 步不下降时，学习率减半，进行精细搜索
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, verbose=True)
criterion = nn.L1Loss()

for epoch in range(8000):
    optimizer.zero_grad()
    
    with torch.no_grad():
        G_x = base_nn(geom_inputs) 
        
# 2. Micro-NN 吐出当前材料的斜率和截距
    gamma_pred, shift_pred = micro_nn(mat_inputs) 
    
    # 3. 仿射变换组装
    dimless_stress_pred = G_x * gamma_pred + shift_pred
    
    # ====================================================
    # 🌟 绝杀修改：在无量纲空间计算 Loss，彻底消灭标度误差！
    # ====================================================
    # 我们不把预测值乘上 s_target 放大，而是把真实误差除以 s_target 缩小！
    # 真实残差 (MPa)
    true_delta = true_fea_stress - mises_theory
    # 无量纲真实残差
    true_dimless_target = true_delta / s_targets
    
    # 强制网络学习：G(x) * gamma + shift == (真值 - 理论) / s
    loss = criterion(dimless_stress_pred, true_dimless_target)
    
    loss.backward()
    optimizer.step()
    # 在优化器 step 之后加上这句，让调度器监控 Loss
    scheduler.step(loss)
    
    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d} | FEA 均方误差 (MSE) Loss: {loss.item():.4f}")

# =====================================================================
# 🌟 模块 6：打印学习成果
# =====================================================================
print("\n🎯 训练完成！Micro-NN 自动推断出的核心材料系数如下：")
micro_nn.eval()
with torch.no_grad():
    gamma_learned, shift_learned = micro_nn(mat_inputs)
    materials_name = ["W", "IN1", "IN3", "OUT3", "CU"]
    
    for i in range(5):
        idx = i * 2
        print(f"[{materials_name[i]}] E={materials_E[i]:.0f}, Alpha={materials_alpha[i]*1e6:.1f} -> "
              f"自动学习 Gamma = {gamma_learned[idx].item():.4f}, "
              f"Shift = {shift_learned[idx].item():.4f}")
        
torch.save(micro_nn.state_dict(), "MICRO-NN-SI-WEIGHTS.pth")