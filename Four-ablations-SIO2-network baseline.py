import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

pi=3.1415926536
L1=26
E1=410000
aa1=4.5e-6
T=100
v1=0.28

L2=25
E2=70000
aa2=6e-7
v2=0.16
h2=0.5

E3=129000
aa3=3e-6
v3=0.28

s=E1*aa1*T/(1-v1)

for j in [90, 80, 70, 60]:
    # ==========================
    # 1. 读取数据
    # ==========================
    data = pd.read_excel("SIO2_Train_Pool_80_sampled.xlsx",sheet_name=f"{j}%")   

    x = data["x"] / 1000
    y = data["y"] / 1000
    z = data["z"] / 1000 
    X = np.column_stack((x, y, z))

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    joblib.dump(scaler_X, f"SIO2_Train_Pool_80_{j}-nromal-full.pkl") 
    y = data[["max"]].values

    # 转 tensor
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # ==========================
    # 2. 定义神经网络
    # ==========================
    class Net(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=256, output_dim=1):
            super(Net, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.model(x)

    net = Net()
    loss_records=[]
    # ==========================
    # ==========================
    # 3. 训练 (直接替换这部分)
    # ==========================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 引入学习率调度器：如果连续 200 次 Loss 不降，学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    EPOCHS = 8000
    loss_records = []
    best_test_loss = float('inf')

    for epoch in range(EPOCHS):
        net.train()
        optimizer.zero_grad()
        outputs = net(X_train)

        # --- 关键改动 1：使用归一化的小数算 Loss，为了让训练稳定 ---
        loss_for_train = criterion(outputs, y_train) 
        loss_for_train.backward()
        # --- 核心新增：梯度裁剪 (稳压器) ---
        # max_norm=1.0 表示如果梯度向量的范数超过 1.0，就强行压缩到 1.0
        # 这样即便遇到 Epoch 2800 那种跳变，模型也会被“按住”，不会乱跑
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 关键改动 2：调度器盯着这个平稳的 Loss 看 ---
        scheduler.step(loss_for_train)

        if (epoch + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                outputs_test = net(X_test)
                
                # --- 关键改动 3：仅在评价和记录时乘以 s，还原成真实应力误差 (MPa) ---
                # 这样你 Excel 里的 Train_Loss 依然是 180 -> 5 这种真实量级
                train_loss_real = criterion(outputs , y_train ).item()
                test_loss_real = criterion(outputs_test , y_test ).item()
                
                
                loss_records.append({
                    'Epoch': epoch + 1,
                    'LR': optimizer.param_groups[0]['lr'],
                    'Train_Loss': train_loss_real,
                    'Test_Loss': test_loss_real
                })
                print(f"Epoch [{epoch+1}/{EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.6f}, Train Loss(Real): {train_loss_real:.4f}, Test Loss(Real): {test_loss_real:.4f}")



    loss_df = pd.DataFrame(loss_records)
    loss_df.to_excel(f"SIO2_Train_Pool_80_{j}-LOSS-normal-full.xlsx", index=False)
    print("损失记录已保存到 loss_records.xlsx")

    # ==========================
    # 4. 保存模型（只保存权重）
    # ==========================
    torch.save(net.state_dict(), f"SIO2_Train_Pool_80_{j}_NET-normal-full.pth")
    print("模型训练完成并已保存为 SIO2_Train_Pool_80_80_NET.pth")


