import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ==========================================
# 1. 物理函数与网络结构定义
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
        return self.model(x)

# ==========================================
# 2. 物理参数与统一验证集预处理
# ==========================================
pi = 3.1415926536
L1=26; E1=410000; aa1=4.1e-6; v1=0.28
L2=25; E2=70000;  aa2=6e-7;   v2=0.16; h2=0.5
E3=129000; aa3=3e-6; v3=0.28
T=125

s = E1 * abs(aa1-aa3) * T / (1 - v1)

# 注意这里的文件名，请确保你的 20% 标准验证集名称对应
val_data = pd.read_excel("W_Validation_Set_20.xlsx")

print("正在提取 W 验证集物理特征...")
X_val_features = []
ideal_mises_list = []
actual_max_list = []

for index, row in val_data.iterrows():
    current_h1 = row["x"] / 1000
    a_val = row["y"] / 1000
    b_val = row["z"] / 1000
    R_current = [current_h1, current_h1 + h2, 5.0]
    
    # 提取第 1 层背景应力
    constants = solve_3layer_cylinder(R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T)
    Sr = constants['A1']
    Sz = E1 * (constants['eps_0'] - aa1 * T) + 2 * v1 * constants['A1']
    
    # Eshelby 计算局部主应力并合成 Mises
    Sigma_z, Sigma_theta = get_eshelby_3d_stress(a_val, b_val, v1, Sr, Sz)
    ideal_mises = np.sqrt(Sigma_z**2 - Sigma_z * Sigma_theta + Sigma_theta**2)
    ideal_mises_list.append(ideal_mises)
    actual_max_list.append(row["max"])
    
    # 组装 W 层特征向量
    a1_feat = a_val / L2
    b1_feat = 1 - b_val / current_h1
    h3 = 5.0 - current_h1 - h2
    k_feat = (current_h1 / h3 * E1 / E3 + 1)**0.5
    n1_feat = ((E2/(2*(1+v2)*E1*h2*current_h1))**0.5)*L2
    
    X_val_features.append([a1_feat, b1_feat, k_feat, n1_feat])

X_val_raw = np.array(X_val_features)
ideal_mises_arr = np.array(ideal_mises_list)
actual_max_arr = np.array(actual_max_list)
print("W 特征提取完成！\n")

# ==========================================
# 3. 循环验证过程
# ==========================================
results_summary = []
origin_boxplot_data = {}
for j in [4, 3, 2, 1]:
    # 1. 加载归一化器，转换验证集
    scaler_X = joblib.load(f"W_Train_Pool_80_10{j}-mech-residual.pkl")
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    # 2. 初始化网络并加载对应权重（注意前缀大小写，你的训练代码存的是 w_Train_...）
    net = ResidualNet()
    net.load_state_dict(torch.load(f"w_Train_Pool_80_10{j}-mech-residual.pth"))
    net.eval()
    
    # 3. 网络预测残差并还原量纲
    with torch.no_grad():
        pred_residual_scaled = net(X_val_tensor).numpy().flatten()
    pred_residual = pred_residual_scaled * s
    
    # 4. 重建最终应力：理论基准 + 神经网络预测的残差
    pred_max = ideal_mises_arr + pred_residual
    
    # 5. 计算评估指标
    mae = np.mean(np.abs(pred_max - actual_max_arr))
    rmse = np.sqrt(np.mean((pred_max - actual_max_arr)**2))
    mape = np.mean(np.abs((pred_max - actual_max_arr) / actual_max_arr)) * 100
    
    results_summary.append({
        'Train_Ratio': f"{j}%", 
        'MAE (MPa)': round(mae, 4), 
        'RMSE (MPa)': round(rmse, 4), 
        'MAPE (%)': round(mape, 4)
    })
    # 🌟 核心：计算每一个样本的绝对相对误差百分比，并存入字典
    relative_errors = np.abs((pred_max - actual_max_arr) / actual_max_arr) * 100
    origin_boxplot_data[f"{j}%"] = relative_errors    
    print(f"--- 训练集比例: {j}% ---")
    print(f"MAE:  {mae:.4f} MPa")
    print(f"RMSE: {rmse:.4f} MPa")
    print(f"MAPE: {mape:.4f} %\n")

# 2. 🌟 保存专供 Origin 画箱线图的宽表
origin_df = pd.DataFrame(origin_boxplot_data)
origin_df.to_excel("W_Origin_mech-residual4.xlsx", index=False)