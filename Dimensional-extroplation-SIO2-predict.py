import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import joblib 
import math
# ==========================================
# 1 & 2. 求解器部分 (拉梅解 & Eshelby，保持原样)
# ==========================================
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

# ==========================================
# 3. 目标预测材料属性 (Cu - 铜)
# ==========================================
pi = 3.1415926536
L2 = 25; T = 125

E1 = 410000; aa1 = 4.5e-6; v1 = 0.28 # 核心已变为铜
E2 = 70000;  aa2 = 6e-7;   v2 = 0.16; h2 = 0.5
E3 = 129000; aa3 = 3e-6;   v3 = 0.28

data = pd.read_excel("WTSV-SIO2-TUOZHAN.xlsx")  
x = data["a"].values / 10
y = data["b"].values / 10 
z = data["c"].values / 100
L1 = data["L"].values / 10
H1 = data["H"].values / 10
# ==========================================
# 4. 加载训练好的几何形网络
# ==========================================
class ResidualNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1):
        super(ResidualNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)  # 🌟 必须与训练时完全一致，无 Softplus！

# 定义我们新训好的材料超网络 (Micro-NN)
class MaterialMicroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Mish(),
            nn.Linear(16, 16), nn.Mish(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        out = self.net(x)
        return out[:, 0], out[:, 1]

model = ResidualNet()
model.load_state_dict(torch.load("W-sio2-GEOM-RESIDUAL.pth")) 
model.eval()

# 2. 载入刚训好的 Micro-NN (预测材料本构)
micro_model = MaterialMicroNet()
# 【注意】请确保你在训练代码最后保存了这个权重文件！
micro_model.load_state_dict(torch.load("MICRO-NN-SIO2-WEIGHTS.pth")) 
micro_model.eval()

scaler_X = joblib.load("w_sio2_geom_features1.pkl")
output_csv = "WTSV-SIO2-TUOZHAN-ANALYTICAL-EXTRAPOLATION1.csv"

# 【极度关键】：这里的归一化必须跟你刚刚那份训练代码完全一致！
# 我注意到你在训练代码里写的是 E_raw / 100000.0，没有减去 129000，这里严格保持同步！
E_norm = E1 / 10000.0      
alpha_norm = aa1 * 1e6                
mat_tensor = torch.tensor([[E_norm, alpha_norm]], dtype=torch.float32)

with torch.no_grad():
    gamma_pred, shift_pred = micro_model(mat_tensor)
    gamma_val = gamma_pred.item()
    shift_val = shift_pred.item()

print(f"✅ Micro-NN 预测成功！推断波幅 Gamma = {gamma_val:.4f}, 平移 Shift = {shift_val:.4f}")



with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["x3(内芯半径)", "x1(长轴)", "x2(短轴)", "理论Mises(MPa)", "预测残差(MPa)", "最终总应力(MPa)"])

    for i in range(len(x)):
        h1 = x[i]
        a = y[i]
        b = z[i]
        L = L1[i]
        H = H1[i]
        # 宏观常数及理论 Mises
        R_current = [h1, h1 + h2, H]
        constants = solve_3layer_cylinder(
            R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
        )
        mises_theory = calculate_mises_stress(
            h1, 2, constants, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T
        )

        Sr_actual = constants['A1']
        Sz_actual = E1 * (constants['eps_0'] - aa1 * T) + 2 * v1 * constants['A1']
        ratio_zr = Sz_actual / (Sr_actual + 1e-9)
        K_z, K_theta = get_eshelby_3d_stress(a, b, v1, 1.0, ratio_zr)

        h3 = H - h1 - h2
        k_val = (h1/h3 * E1/E3 + 1)**0.5
        n1_val = ((E2/(2*(1+v2)*E1*h2*h1))**0.5)*L
        
        # 将Cu的几何特征传入网络，提取无量纲波动
        inp_raw = np.array([[K_theta, K_z, k_val, n1_val]])
        inp_scaled = scaler_X.transform(inp_raw)
        inp = torch.tensor(inp_scaled, dtype=torch.float32)

        with torch.no_grad():
            delta_dimless_W = model(inp).item()
            
        # ====================================================
        # 4. 完美闭环：用新网络推测出的 Gamma 和 Shift 进行修正
        # ====================================================

        delta_dimless_target = (delta_dimless_W * gamma_val) + shift_val
        
        s_target = E1 * abs(aa1 - 3e-6) * T / (1 - v1)
        delta_val_real = delta_dimless_target * s_target
        y_pred_real = mises_theory + delta_val_real
        
        writer.writerow([h1, a, b, mises_theory, delta_val_real, y_pred_real])

print("✔ 铜(Cu)层解析物理外推预测完成，结果已写入：", output_csv)