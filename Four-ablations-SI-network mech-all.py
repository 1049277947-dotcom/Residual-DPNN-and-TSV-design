import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 🌟 1. 理论基准求解器 (3层广义平面应变)
# ==========================================
def solve_3layer_cylinder(R, E, nu, alpha, dT, outer_bc='fixed'):
    R1, R2, R3 = R
    E1, E2, E3 = E
    n1, n2, n3 = nu
    a1, a2, a3 = alpha
    
    V1, V2, V3 = R1**2, R2**2 - R1**2, R3**2 - R2**2
    K, F = np.zeros((6, 6)), np.zeros(6)
    
    K[0,0], K[0,1], K[0,2] = 1.0, -1.0, 1.0/(R1**2)
    K[1,0], K[1,1], K[1,2], K[1,5] = (1+n1)*(1-2*n1)/E1, -(1+n2)*(1-2*n2)/E2, -(1+n2)/(E2*R1**2), -(n1-n2)
    F[1] = (1+n2)*a2*dT - (1+n1)*a1*dT
    
    K[2,1], K[2,2], K[2,3], K[2,4] = 1.0, -1.0/(R2**2), -1.0, 1.0/(R2**2)
    K[3,1], K[3,2], K[3,3], K[3,4], K[3,5] = (1+n2)*(1-2*n2)/E2, (1+n2)/(E2*R2**2), -(1+n3)*(1-2*n3)/E3, -(1+n3)/(E3*R2**2), -(n2-n3)
    F[3] = (1+n3)*a3*dT - (1+n2)*a2*dT
    
    if outer_bc == 'fixed':
        K[4,3], K[4,4], K[4,5] = (1+n3)*(1-2*n3)/E3, (1+n3)/(E3*R3**2), -n3
        F[4] = -(1+n3)*a3*dT
        
    K[5,0], K[5,1], K[5,3], K[5,5] = 2*n1*V1, 2*n2*V2, 2*n3*V3, E1*V1 + E2*V2 + E3*V3
    F[5] = (E1*V1*a1 + E2*V2*a2 + E3*V3*a3)*dT
    
    X = np.linalg.solve(K, F)
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

# ==========================================
# 🌟 2. Eshelby 大一统 3D 应力影响因子求解器
# ==========================================
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
for j in [50, 40, 30, 20, 10]:
    # ==========================
    # 3. 物理参数定义与数据读取
    # ==========================
    pi = 3.1415926536
    L1 = 26; L2 = 25; T = 125
    E1 = 410000; aa1 = 4.5e-6; v1 = 0.28
    E2 = 70000;  aa2 = 6e-7;   v2 = 0.16; h2 = 0.5
    E3 = 129000; aa3 = 3e-6;   v3 = 0.28

    # 全局尺度因子 s
    s = E1 * abs(aa1 - aa3) * T / (1 - v1)

    data = pd.read_excel("SI_Train_Pool_80_sampled2.xlsx",sheet_name=f"{j}%") 

    # ==========================================
    # 4. 核心逻辑：特征提取与理论残差计算
    # ==========================================
    X_features = []
    theory_mises_list = []
    residual_list = []

    for index, row in data.iterrows():
        current_h1 = row["x"] / 1000
        a_val = row["y"] / 1000
        b_val = row["z"] / 1000
        
        R_current = [current_h1, current_h1 + h2, 5.0]
        
        # 求解宏观 3 层套筒背景常数
        constants = solve_3layer_cylinder(
            R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
        )
        
        # 针对 Si 层内边界计算理论 Mises 应力
        mises_theory = calculate_mises_stress(
            current_h1 + h2, 3, constants, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T
        )
        theory_mises_list.append(mises_theory)
        residual_list.append(row["max"])
        
        # 提取 Si 层真实宏观双轴应力以计算【激励比例】
        Sr_actual = constants['A3']
        Sz_actual = E3 * (constants['eps_0'] - aa3 * T) + 2 * v3 * constants['A3']
        ratio_zr = Sz_actual / (Sr_actual + 1e-9)
        
        # 调用 Eshelby 提取 3D 无量纲影响因子
        K_z, K_theta = get_eshelby_3d_stress(a_val, b_val, v3, 1.0, ratio_zr)
        
        # 组装 4 维物理特征
        h3 = 5.0 - current_h1 - h2
        k_val = (h3/current_h1 * E3/E1 + 1)**0.5
        n1_val = ((E2/(2*(1+v2)*E3*h2*h3))**0.5)*L2
        
        X_features.append([K_theta, K_z, k_val, n1_val])

    X = np.array(X_features)
    data["theory_mises"] = theory_mises_list
    data["residual_stress"] = residual_list

    # ==========================================
    # 5. X 和 Y 双向标准化
    # ==========================================
    # 原始无量纲目标值 Y
    y_raw = (data["residual_stress"] / s).values.reshape(-1, 1)

    # 对特征 X 进行标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)


    joblib.dump(scaler_X, f"SI_Train_Pool_80_{j}-mech-all.pkl")

    # 划分数据集 (使用缩放后的 X_scaled 和 y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw , test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # ==========================
    # 6. 定义残差网络 
    # ==========================
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

    net = ResidualNet()

    # ==========================================
    # 7. 训练过程 (一路到底，只存最终版)
    # ==========================================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    EPOCHS = 8000
    loss_records = []
    final_test_error = 0.0

    for epoch in range(EPOCHS):
        net.train()
        optimizer.zero_grad()
        outputs = net(X_train)
        
        # 网络在标准化空间内极速学习
        loss_for_train = criterion(outputs, y_train) 
        loss_for_train.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_for_train)

        # 每 50 轮进行真实物理误差评估并打印
        if (epoch + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                outputs_test = net(X_test)
                train_loss_real = criterion(outputs * s, y_train * s).item()
                test_loss_real = criterion(outputs_test * s, y_test * s).item()
                final_test_error = np.sqrt(test_loss_real)
                
                loss_records.append({
                    'Epoch': epoch + 1,
                    'LR': optimizer.param_groups[0]['lr'],
                    'Train_Residual_Loss(MPa)': np.sqrt(train_loss_real),
                    'Test_Residual_Loss(MPa)': final_test_error
                })
                
                print(f"Epoch [{epoch+1}/{EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.6f}, Train Error: {np.sqrt(train_loss_real):.4f} MPa, Test Error: {final_test_error:.4f} MPa")


    # ==========================================
    # 6. 保存模型
    # ==========================================
    torch.save(net.state_dict(), f"SI_Train_Pool_80_{j}-mech-all.pth") 
    pd.DataFrame(loss_records).to_excel(f"SI_Train_Pool_80_{j}-mech-all-LOSS.xlsx", index=False)
    print(f"\n✔ 几何特征网络训练完成！最终测试误差: {final_test_error:.4f} MPa")

    ##  用来测试训练损失曲线的