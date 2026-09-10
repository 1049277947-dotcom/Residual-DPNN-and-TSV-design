import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
# ==========================================
# 1. 定义与训练代码完全一致的神经网络
# ==========================================
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

# ==========================================
# 2. 读取 20% 标准验证集并预处理
# ==========================================
# 确保你之前划分的 20% 验证集文件名为这个
val_data = pd.read_excel("SI_Validation_Set_20.xlsx")

# 按照训练代码的处理方式：仅除以 1000
x_val = val_data["x"] / 1000
y_val = val_data["y"] / 1000
z_val = val_data["z"] / 1000
X_val_raw = np.column_stack((x_val, y_val, z_val))

# 提取真实的 Max 和 Ave 目标值
y_true_max = val_data["max"].values
y_true_ave = val_data["ave"].values

# ==========================================
# 3. 循环调用各个比例的模型进行验证
# ==========================================
results_summary = []
ratios = [90, 80, 70, 60]
# 🌟 新增：专门用于收集 Origin 绘图数据的字典
origin_boxplot_data = {}
for j in ratios:
    print(f"================ 开始验证 {j}% 模型 ================")
    # 1. 加载归一化器，转换验证集
    scaler_X = joblib.load(f"SI_Train_Pool_80_{j}-nromal-full.pkl")
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)      
    # 1. 初始化网络并加载对应比例的权重
    net = Net()
    model_path = f"SI_Train_Pool_80_{j}_NET-nromal-full.pth" 
    
    try:
        net.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"⚠️ 找不到模型文件 {model_path}，请检查该比例是否已训练！跳过此比例。\n")
        continue
        
    net.eval()
    
    # 2. 进行预测
    with torch.no_grad():
        predictions = net(X_val_tensor).numpy()
        
    # 神经网络输出了两个值，第一列是 max，第二列是 ave
    pred_max = predictions[:, 0]

    
    # 3. 计算 Max 的误差指标
    mae_max = np.mean(np.abs(pred_max - y_true_max))
    rmse_max = np.sqrt(np.mean((pred_max - y_true_max)**2))
    mape_max = np.mean(np.abs((pred_max - y_true_max) / y_true_max)) * 100
    

    
    # 记录结果
    results_summary.append({
        'Train_Ratio': f"{j}%", 
        'Max_MAE (MPa)': round(mae_max, 4), 
        'Max_RMSE (MPa)': round(rmse_max, 4), 
        'Max_MAPE (%)': round(mape_max, 4),

    })
    # 🌟 核心：计算每一个样本的绝对相对误差百分比，并存入字典
    relative_errors = np.abs((pred_max - y_true_max) / y_true_max) * 100
    origin_boxplot_data[f"{j}%"] = relative_errors    
    print(f"【Max 应力】 MAE: {mae_max:.4f} | RMSE: {rmse_max:.4f} | MAPE: {mape_max:.4f}%")


# 2. 🌟 保存专供 Origin 画箱线图的宽表
origin_df = pd.DataFrame(origin_boxplot_data)
origin_df.to_excel("SI_Origin-normal-full.xlsx", index=False)