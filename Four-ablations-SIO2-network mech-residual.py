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
# 1. 理论基准求解器 (保持不变)
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

for j in [4, 3, 2, 1]:
    # ==========================
    # 2. 物理参数定义与数据读取 (W - 钨)
    # ==========================
    pi = 3.1415926536
    L1=26; E1=410000; aa1=4.1e-6; v1=0.28
    L2=25; E2=70000;  aa2=6e-7;   v2=0.16; h2=0.5
    E3=129000; aa3=3e-6; v3=0.28
    T=125

    # 🌟 加上了绝对值，保证标度为正！
    s = E1 * abs(aa1-aa3) * T / (1 - v1)
    #s = E1 * aa1 * T / (1 - v1)-E3*aa3* T / (1 - v3)
    data = pd.read_excel(f"SIO2_Train_Pool_80_10{j}.xlsx",sheet_name="10%") 

    # ==========================================
    # 3. 特征提取与残差目标构建
    # ==========================================
    X_features = []
    residual_list = []

    for index, row in data.iterrows():
        current_h1 = row["x"] / 1000
        a_val = row["y"] / 1000
        b_val = row["z"] / 1000
        
        R_current = [current_h1, current_h1 + h2, 5.0]
        constants = solve_3layer_cylinder(
            R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
        )
        
        mises_theory = calculate_mises_stress(
            current_h1, 2, constants, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T
        )
        # 纯残差目标 (W材料允许有负数)
        residual_list.append(row["max"] - mises_theory) 
        
        Sr_actual = constants['A1']
        Sz_actual = E1 * (constants['eps_0'] - aa1 * T) + 2 * v1 * constants['A1']
        ratio_zr = Sz_actual / (Sr_actual + 1e-9)
        K_z, K_theta = get_eshelby_3d_stress(a_val, b_val, v1, 1.0, ratio_zr)
        
        h3 = 5.0 - current_h1 - h2
        k_val = (current_h1/h3 * E1/E3 + 1)**0.5
        n1_val = ((E2/(2*(1+v2)*E1*h2*current_h1))**0.5)*L2
        
        X_features.append([K_theta, K_z, k_val, n1_val])

    X = np.array(X_features)
    y = (np.array(residual_list) / s).reshape(-1, 1)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, f"SIO2_Train_Pool_80_10{j}-mech-residual.pkl") 

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # ==========================
    # 4. 定义残差网络 (🌟 彻底拆除所有限制，还原最强非线性拟合)
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
            return self.model(x)  # 直接输出，自由拟合正负数

    net = ResidualNet()

    # ==========================================
    # 5. 纯数据驱动训练过程
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
        
        outputs_residual = net(X_train)
        # 🌟 唯一目标：让 MSE 降到极低！
        loss_for_train = criterion(outputs_residual, y_train) 
        
        loss_for_train.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
        optimizer.step()
        scheduler.step(loss_for_train)

        if (epoch + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                outputs_test = net(X_test)
                train_loss_real = criterion(outputs_residual * s, y_train * s).item()
                test_loss_real = criterion(outputs_test * s, y_test * s).item()
                final_test_error = np.sqrt(test_loss_real)
                
                loss_records.append({
                    'Epoch': epoch + 1,
                    'LR': optimizer.param_groups[0]['lr'],
                    'Train_Loss': train_loss_real,
                    'Test_Loss': test_loss_real
                })
                print(f"Epoch [{epoch+1}/{EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.6f}, Train Loss(Real): {train_loss_real:.4f}, Test Loss(Real): {test_loss_real:.4f}")

    # ==========================================
    # ==========================================
    # 6. 保存模型
    # ==========================================
    torch.save(net.state_dict(), f"SIO2_Train_Pool_80_10{j}-mech-residual.pth") 
    pd.DataFrame(loss_records).to_excel(f"SIO2_Train_Pool_80_10{j}-mech-residual-LOSS.xlsx", index=False)
    print(f"\n✔ 几何特征网络训练完成！最终测试误差: {final_test_error:.4f} MPa")