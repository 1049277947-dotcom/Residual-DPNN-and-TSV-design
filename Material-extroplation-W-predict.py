import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import joblib 
import math

# ==========================================
# 1. 求解器部分 (保持原样)
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
# 2. 目标预测材料属性 (Cu - 铜)
# ==========================================
pi = 3.1415926536
L2 = 25; T = 125

E1 = 410000; aa1 = 4.1e-6; v1 = 0.28  # 目标预测金属为铜
E2 = 70000; aa2 = 6e-7;   v2 = 0.16; h2 = 0.5
E3 = 129000; aa3 = 3e-6;  v3 = 0.28

data = pd.read_excel("W-W.xlsx")  
x = data["x"].values / 10
y = data["y"].values  
z = data["z"].values / 10

# ==========================================
# 3. 网络架构定义
# ==========================================
class ResidualNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1): # 🌟 保持 4 维基准波形提取
        super(ResidualNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class MaterialMicroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.Mish(),  # 🌟 5 维输入: [E, alpha] + [a, b, h1]
            nn.Linear(16, 16), nn.Mish(),
            nn.Linear(16, 2)
        )
    def forward(self, mat_features, geom_dims):
        # 🌟 拼接：将全局材料属性与当前孔洞尺寸融合
        x = torch.cat([mat_features, geom_dims], dim=1)
        out = self.net(x)
        return out[:, 0], out[:, 1]

# ==========================================
# 4. 加载权重与前置处理
# ==========================================
model = ResidualNet()
model.load_state_dict(torch.load("W-W-GEOM-RESIDUAL1.pth")) 
model.eval()

micro_model = MaterialMicroNet()
micro_model.load_state_dict(torch.load("MICRO-NN-W-WEIGHTS1.pth")) 
micro_model.eval()

scaler_X = joblib.load("scaler_w_W_geom_features1.pkl")
output_csv = "W-W-ANALYTICAL-EXTRAPOLATION1.csv"

# 🌟 材料属性全局归一化，放在循环外部只做一次
E_norm = E1 / 10000.0      
alpha_norm = aa1 * 1e6                
mat_tensor = torch.tensor([[E_norm, alpha_norm]], dtype=torch.float32)

# ==========================================
# 5. 循环预测全流程
# ==========================================
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["x3(内芯半径)", "x1(长轴)", "x2(短轴)", "理论Mises(MPa)", "预测残差(MPa)", "最终总应力(MPa)"])

    for i in range(len(x)):
        h1 = x[i]; a = y[i]; b = z[i]
        
        # --- A. 计算解析解基准 ---
        R_current = [h1, h1 + h2, 5.0]
        constants = solve_3layer_cylinder(
            R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
        )

        Sr_actual = constants['A1']
        Sz_actual = E1 * (constants['eps_0'] - aa1 * T) + 2 * v1 * constants['A1']

        Sigma_z, Sigma_theta = get_eshelby_3d_stress(a, b, v1, Sr_actual, Sz_actual)
        ideal_mises = np.sqrt(Sigma_z**2 - Sigma_z * Sigma_theta + Sigma_theta**2)
        
        # --- B. 计算特征标度和无量纲化 ---
        s_target = E1 * abs(aa1 - 3e-6) * T / (1 - v1)
        h3 = 5.0 - h1 - h2
        k_val = (h1/h3 * E1/E3 + 1)**0.5
        n1_val = ((E2/(2*(1+v2)*E1*h2*h1))**0.5)*L2
        ss=h1/5
        a1 = a / L2
        b1 = 1 - b/h1
        
        # --- C. 阶段 1：获取 Base-NN 的 4D 基准波形 ---
        inp_raw = np.array([[ss,a1, b1, k_val, n1_val]])
        inp_scaled = scaler_X.transform(inp_raw)
        inp = torch.tensor(inp_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            delta_dimless_W = model(inp).item()

        # --- D. 阶段 2：获取 Micro-NN 考虑了具体尺寸的动态仿射系数 ---
        # 🌟 把当前孔洞的 a, b, h1 转为 [1, 3] 维 Tensor 参与拼接组合
        geom_dims_tensor = torch.tensor([[ h1]], dtype=torch.float32)

        with torch.no_grad():
            gamma_pred, shift_pred = micro_model(mat_tensor, geom_dims_tensor)
            gamma_val = gamma_pred.item()
            shift_val = shift_pred.item()

        # --- E. 阶段 3：组装并反归一化 ---
        delta_dimless_target = (delta_dimless_W * gamma_val) + shift_val
        delta_val_real = delta_dimless_target * s_target
        y_pred_real = ideal_mises + delta_val_real
        
        writer.writerow([h1, a, b, ideal_mises, delta_val_real, y_pred_real])
        
        # 实时打印监控动态 Gamma 的变化

        print(f"进程 [{i}/{len(x)}] -> a={a:.2f}, b={b:.2f} | 局部 Gamma: {gamma_val:.4f}, Shift: {shift_val:.4f}")

print("\n✔ 金属(Cu)层【尺寸动态约束】预测完成，结果已写入：", output_csv)