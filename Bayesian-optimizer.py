import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import colors
from math import pi
import itertools
import pandas as pd
import torch
import torch.nn as nn
import joblib
import torch.optim as optim
from sklearn.model_selection import train_test_split
import csv

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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16        # 全局默认字体大小

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

s1=E1*aa1*T/(1-v1)
s2=E2*aa2*T/(1-v2)
s3=E3*aa3*T/(1-v3)
# --------------------------
# 1) 模型与预测函数
# --------------------------
# 4. 定义残差网络 (🌟 彻底拆除所有限制，还原最强非线性拟合)
# ==========================
class ResidualNetW(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1):
        super(ResidualNetW, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)  # 直接输出，自由拟合正负数

class ResidualNetSIO2(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1):
        super(ResidualNetSIO2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)  # 直接输出，自由拟合正负数

class ResidualNetSI(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1):
        super(ResidualNetSI, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)  # 直接输出，自由拟合正负数


def load_modelW(model_path):
    model = ResidualNetW()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_modelSIO2(model_path):
    model = ResidualNetSIO2()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_modelSI(model_path):
    model = ResidualNetSI()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
# 只加载一次！

model_w    = load_modelW("MICRO-NN-W-WEIGHTS1.pth")
model_sio2 = load_modelSIO2("W-GEOM-RESIDUAL.pth")
model_si   = load_modelSI("W-SI-GEOM-RESIDUAL.pth")

scaler_W = joblib.load("scaler_w_W_geom_features2.pkl")
scaler_SIO2 = joblib.load("w_sio2_geom_features1.pkl")
scaler_SI = joblib.load("w_SI_geom_features1.pkl")

def predict_batchW(model, x,y,z):
    """xyz_np: shape (N,3) 的 numpy 数组"""
    h1=x
    a=y 
    b=z
    R_current = [h1, h1 + h2, 5]
    constants = solve_3layer_cylinder(
        R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
    )

    Sr_actual = constants['A1']
    Sz_actual = E1 * (constants['eps_0'] - aa1 * T) + 2 * v1 * constants['A1']

    Sigma_z, Sigma_theta = get_eshelby_3d_stress(a, b, v1, Sr_actual, Sz_actual)
    ideal_mises = np.sqrt(Sigma_z**2 - Sigma_z * Sigma_theta + Sigma_theta**2)
    
    # --- B. 计算特征标度和无量纲化 ---
    s_target = E1 * abs(aa1 - 3e-6) * T / (1 - v1)
    h3 = 5 - h1 - h2
    k_val = (h1/h3 * E1/E3 + 1)**0.5
    n1_val = ((E2/(2*(1+v2)*E1*h2*h1))**0.5)*25
    ss=h1/5
    a1 = a / 25
    b1 = 1 - b/h1
    
    # --- C. 阶段 1：获取 Base-NN 的 4D 基准波形 ---
    inp_raw = np.array([[a1, b1, k_val, n1_val]])
    inp_scaled = scaler_W.transform(inp_raw)
    inp = torch.tensor(inp_scaled, dtype=torch.float32)
    with torch.no_grad():
        delta_dimless_W = model(inp).item()
    delta_val_real = delta_dimless_W * s_target
    y_pred_real = ideal_mises + delta_val_real
    return y_pred_real # [:,0], [:,1]

def predict_batchSIO2(model, x,y,z):
    """xyz_np: shape (N,3) 的 numpy 数组"""
    h1=x
    a=y 
    b=z                                                #对于是否引入椭球部分无量纲数，实验引入t后效果不明显，约为0.05
    R_current = [h1, h1 + h2, 5.0]
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

    h3 = 5.0 - h1 - h2
    k_val = (h1/h3 * E1/E3 + 1)**0.5
    n1_val = ((E2/(2*(1+v2)*E1*h2*h1))**0.5)*L2
    
    # 将Cu的几何特征传入网络，提取无量纲波动
    inp_raw = np.array([[K_theta, K_z, k_val, n1_val]])
    inp_scaled = scaler_SIO2.transform(inp_raw)
    inp = torch.tensor(inp_scaled, dtype=torch.float32)
    with torch.no_grad():
        delta_dimless_W = model(inp).item()
        
    # ====================================================
    # 4. 完美闭环：用新网络推测出的 Gamma 和 Shift 进行修正
    # ====================================================
    s_target = E1 * abs(aa1 - 3e-6) * T / (1 - v1)
    delta_val_real = delta_dimless_W * s_target
    y_pred_real = mises_theory + delta_val_real
    return y_pred_real

def predict_batchSI(model, x,y,z):
    """xyz_np: shape (N,3) 的 numpy 数组"""
    h1=x
    a=y 
    b=z
    R_current = [h1, h1 + h2, 5]
    constants = solve_3layer_cylinder(
        R_current, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T, outer_bc='fixed'
    )
    mises_theory = calculate_mises_stress(
        h1 + h2, 3, constants, [E1, E2, E3], [v1, v2, v3], [aa1, aa2, aa3], T
    )

    Sr_actual = constants['A3']
    Sz_actual = E3 * (constants['eps_0'] - aa3 * T) + 2 * v3 * constants['A3']
    ratio_zr = Sz_actual / (Sr_actual + 1e-9)
    K_z, K_theta = get_eshelby_3d_stress(a, b, v3, 1.0, ratio_zr)

    h3 = 5 - h1 - h2
    k_val = (h3/h1 * E3/E1 + 1)**0.5
    n1_val = ((E2/(2*(1+v2)*E3*h2*h3))**0.5)*25
    
    # 将Cu的几何特征传入网络，提取无量纲波动
    inp_raw = np.array([[K_theta, K_z, k_val, n1_val]])
    inp_scaled = scaler_SI.transform(inp_raw)
    inp = torch.tensor(inp_scaled, dtype=torch.float32)
    with torch.no_grad():
        delta_dimless_W = model(inp).item()
        
    # ====================================================
    # 4. 完美闭环：用新网络推测出的 Gamma 和 Shift 进行修正
    # ====================================================
    s_target = E1 * abs(aa1 - 3e-6) * T / (1 - v1)
    delta_val_real = delta_dimless_W * s_target
    y_pred_real = mises_theory + delta_val_real
    return y_pred_real
# -----------------------------
# 你的 compute_resistance 函数
# -----------------------------
def compute_resistance(rho, a, L, b, c):
    eps = 1e-12

    def integrand(z):
        # 原函数中如果分母产生负值，用 eps 替代
        den_val = a**2 - c**2*(1 - z**2/b**2)
        if den_val == 0:
            den_val = eps
        # integrand = 1/den_val (你原来写的似乎返回 den 而不是 1/den？我保留你原意：integrand = 1/(a^2 - ...)
        val = 1.0 / den_val
        return val

    integral_result, _ = quad(integrand, -b, b, limit=200)
    resistance = (rho / (pi * a**2)) * (L - 2.0*b) + rho / pi * integral_result
    return resistance

# -----------------------------
# 你的 w 计算（请替换为真实公式）
# -----------------------------
def compute_w(x, y, z):
    f1_w   = predict_batchW(model_w,x,y,z)
    f1_sio2 = predict_batchSIO2(model_sio2,x,y,z)
    f1_si  = predict_batchSI(model_si, x,y,z)
    rr = (
        #1-(1-f1_w/ w_max)*(1-f1_sio2/sio2_max)*(1-f1_si/si_max )                                                                                              #这个一般般
        #w1*(f1_w/ w_max)**2   + w2*(f2_w/ w_ave)**2  + w3*(f1_sio2/sio2_max)**2  + w4*(f2_sio2/sio2_ave)**2  + w5*(f1_si/si_max)**2  + w6*(f2_si/si_ave)**2   #这个不错
    #w1*f1_w/ w_max   + w2*f2_w/ w_ave + w3*f1_sio2/sio2_max + w4*f2_sio2/sio2_ave + w5*f1_si/si_max     + w6*f2_si/si_ave 
    #(f1_w/ w_max)**2 + (f1_sio2/sio2_max)**2 + (f1_si/si_max)**2 - f1_w/ w_max*f1_sio2/sio2_max- f1_w/ w_max*f1_si/si_max -f1_sio2/sio2_max*f1_si/si_max
    #np.maximum.reduce([f1_w / w_max,f1_sio2 / sio2_max,f1_si / si_max])
    1/100*np.log(np.exp(100*f1_w/ w_max)+np.exp(100*f1_sio2/sio2_max)+np.exp(100*f1_si/si_max))
)
    return rr

# ========== 参数空间 ==========
a_vals = np.linspace(5, 25.0, 81)
b_vals = np.linspace(0.1, 0.95, 41)
h_vals = np.linspace(1.0, 2.0, 41)

A, B, H = np.meshgrid(a_vals, b_vals, h_vals, indexing='ij')


Nh = 41   
Na = 81   
Nb = 41   
rho = 5.28e-2
L   = 52.0
w1=0
w2=0
w3=1/3
w4=0
w5=1/3
w6=0

c   = 0.5


res_min = rho*52/1.5/1.5/pi
all_ave = 81.80525618

w_max=108.381; w_ave=94.77238281
sio2_max=136.054; sio2_ave=99.44784865
si_max=72.4717;   si_ave=67.36018739

w_max=980
sio2_max=360
si_max=500
# 约束阈值
R_th = res_min * 1.5

# ========== 接口 ==========
def R_func(x):
    h, a, b = x
    return compute_resistance(rho, h, L, a, b)

def W_func(x):
    h, a, b = x
    return compute_w(h, a, b)

print("Evaluating R over full grid...")

R_grid = np.zeros_like(A)
for i in range(H.shape[0]):
    for j in range(A.shape[1]):
        for k in range(B.shape[2]):
            R_grid[i, j, k] = R_func([H[i,j,k], A[i,j,k], B[i,j,k]])

feasible_mask = R_grid < R_th

# 所有可行点（真正的连续区域）
feasible_points = np.column_stack([
    H[feasible_mask],
    A[feasible_mask],
    B[feasible_mask]
])

print(f"Feasible points: {len(feasible_points)}")

# ========== GP 模型 ==========
kernel = (
    ConstantKernel(1.0, (1e-2, 1e2))
    * Matern(length_scale=0.5, nu=2.5)
    + WhiteKernel(1e-6)
)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

# ========== EI ==========
def expected_improvement(x, gp, y_best):
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)



# ========== BO ==========
n_init = 1
n_iter = 100
n_cand = 1800

# ========== 人为指定起始点 ==========
x0 = np.array([2.0, 1, 0.11])

# 安全检查（非常重要，论文也讲得通）
assert R_func(x0) < R_th, "Initial point is not feasible!"

# 其余初始点仍然随机
X_rand = feasible_points[
    np.random.choice(len(feasible_points), n_init-1, replace=False)
]

X = np.vstack([x0, X_rand])
yW = np.array([W_func(x) for x in X])

# 初始点：只从可行域里选
#X = feasible_points[
#    np.random.choice(len(feasible_points), n_init, replace=False)
#]
#yW = np.array([W_func(x) for x in X])
X_history = [x0]
W_history = [W_func(x0)]
R_history = [R_func(x0)]
history = []

for it in range(n_iter):
    gp.fit(X, yW)

    y_best = np.min(yW)
    shrink = max(0.2, 1.0 - it / n_iter)   # 后期不会小于 20%
    # ⭐ 局部采样（路径不会乱跳）
    center = X[np.argmin(yW)]
    step = np.array([
    0.03 * (h_vals.max() - h_vals.min()),   # h
    0.03 * (a_vals.max() - a_vals.min()),   # a
    0.03 * (b_vals.max() - b_vals.min()),   # b
    ])

    cand = center + np.random.randn(n_cand, 3) * step


    # 参数边界
    cand = cand[
        (cand[:,0] >= h_vals.min()) & (cand[:,0] <= h_vals.max()) &
        (cand[:,1] >= a_vals.min()) & (cand[:,1] <= a_vals.max()) &
        (cand[:,2] >= b_vals.min()) & (cand[:,2] <= b_vals.max())
    ]

    # ⭐ 核心：直接用 R 判断可行性
    cand_feasible = []
    for x in cand:
        if R_func(x) < R_th:
            cand_feasible.append(x)

    cand_feasible = np.array(cand_feasible)

    # 兜底（防止极端情况）
    if len(cand_feasible) == 0:
        cand_feasible = feasible_points[
            np.random.choice(len(feasible_points), n_cand, replace=False)
        ]

    acq = np.array([
    expected_improvement(x, gp, y_best)
    for x in cand_feasible
    ])

    x_next = cand_feasible[np.argmax(acq)]
    w_next = W_func(x_next)
    r_next = R_func(x_next)

    X_history.append(x_next)
    W_history.append(w_next)
    R_history.append(r_next)

    X = np.vstack([X, x_next])
    yW = np.append(yW, W_func(x_next))


    print(f"Iter {it+1:02d} | Best w = {np.min(yW):.4f}")

from skimage import measure


a_lin = np.linspace(0.01, 25.0, Na)
b_lin = np.linspace(0.01, 0.95, Nb)
h_lin = np.linspace(1.0, 2.0, Nh)

A, B, H = np.meshgrid(a_lin, b_lin, h_lin, indexing="ij")

R_grid = np.zeros_like(A)

for i in range(Na):
    for j in range(Nb):
        for k in range(Nh):
            R_grid[i,j,k] = R_func([H[i,j,k], A[i,j,k], B[i,j,k]])
verts, faces, _, _ = measure.marching_cubes(
    R_grid, level=R_th
)
verts_phys = np.zeros_like(verts)
verts_phys[:, 0] = a_lin[0] + verts[:, 0] / (Na - 1) * (a_lin[-1] - a_lin[0])
verts_phys[:, 1] = b_lin[0] + verts[:, 1] / (Nb - 1) * (b_lin[-1] - b_lin[0])
verts_phys[:, 2] = h_lin[0] + verts[:, 2] / (Nh - 1) * (h_lin[-1] - h_lin[0])


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

mesh = Poly3DCollection(verts_phys[faces], alpha=0.25)
mesh.set_facecolor((1.0, 0.6, 0.7))  # 粉色
mesh.set_edgecolor("none")

ax.add_collection3d(mesh)
X_path = np.array(X_history)   # shape (N,3)
W_path = np.array(W_history)
from matplotlib import cm
from matplotlib.colors import Normalize

norm = Normalize(vmin=W_path.min(), vmax=W_path.max())
cmap = cm.viridis

for i in range(len(X_path)-1):
    ax.plot(
        X_path[i:i+2,1],  # a
        X_path[i:i+2,2],  # b
        X_path[i:i+2,0],  # h
        color=cmap(norm(W_path[i])),
        linewidth=2
    )
ax.scatter(
    X_path[0,1], X_path[0,2], X_path[0,0],
    color="green", s=80, label="Start"
)

best_id = np.argmin(W_path)
ax.scatter(
    X_path[best_id,1],
    X_path[best_id,2],
    X_path[best_id,0],
    color="red", s=120, marker="*", label="Best"
)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.6, label="w")

offset_a = 0.1 * (a_vals.max() - a_vals.min())
offset_b = 0.1 * (b_vals.max() - b_vals.min())
offset_h = 0.03 * (h_vals.max() - h_vals.min())


h0 = h_vals.min()
a0 = a_vals.max() + offset_a   # 右墙向右挪
b0 = b_vals.max() + offset_b   # 前墙向前挪
# ===============================
# 3D 中的正交投影路径
# ===============================

# 投影到 a-b 平面 (h = h0)
ax.plot(
    X_path[:, 1], X_path[:, 2], np.full_like(X_path[:, 0], h0),
    linestyle="--", color="gray", linewidth=1, alpha=0.6
)

# 投影到 a-h 平面 (b = b0)
ax.plot(
    X_path[:, 1], np.full_like(X_path[:, 1], b0), X_path[:, 0],
    linestyle="--", color="gray", linewidth=1, alpha=0.6
)

# 投影到 b-h 平面 (a = a0)
ax.plot(
    np.full_like(X_path[:, 2], a0), X_path[:, 2], X_path[:, 0],
    linestyle="--", color="gray", linewidth=1, alpha=0.6
)
ax.scatter(
    X_path[:, 1], X_path[:, 2], np.full_like(X_path[:, 0], h0),
    c=W_path, cmap="viridis",
    s=12, alpha=0.5
)

ax.scatter(
    X_path[:, 1], np.full_like(X_path[:, 2], b0), X_path[:, 0],
    c=W_path, cmap="viridis",
    s=12, alpha=0.5
)

ax.scatter(
    np.full_like(X_path[:, 1], a0), X_path[:, 2], X_path[:, 0],
    c=W_path, cmap="viridis",
    s=12, alpha=0.5
)
plt.show()










fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

proj_pairs = [
    (1, 2, "a", "b"),
    (1, 0, "a", "h"),
    (2, 0, "b", "h"),
]

for ax, (i, j, xi, yj) in zip(axes, proj_pairs):


    # 优化路径投影
    sc = ax.scatter(
        X_path[:, i],
        X_path[:, j],
        c=W_path,
        cmap="viridis",
        s=25,
        edgecolors="k"
    )

    ax.plot(
        X_path[:, i],
        X_path[:, j],
        color="gray",
        linewidth=1,
        alpha=0.7
    )

    # 起点 & 终点
    ax.scatter(
        X_path[0, i], X_path[0, j],
        color="green", s=60, label="Start"
    )
    ax.scatter(
        X_path[best_id, i], X_path[best_id, j],
        color="red", marker="*", s=120, label="Best"
    )

    # 画 R 可行域投影（淡粉）
    ax.scatter(
        feasible_points[:, i],
        feasible_points[:, j],
        s=3,
        color=(1.0, 0.75, 0.8),
        alpha=0.15,
        label="Feasible region"
    )

    ax.set_xlabel(xi)
    ax.set_ylabel(yj)
    ax.set_aspect("auto")


plt.tight_layout()
plt.show()


# ====== 沿路径计算应力 ======
W_max_list, W_ave_list = [], []
SiO2_max_list, SiO2_ave_list = [], []
Si_max_list, Si_ave_list = [], []

for x in X_path:
    h, a, b = x
    f1_w,   f2_w    = predict_batchW(model_w,h,a,b)
    f1_sio2,f2_sio2  = predict_batchSIO2(model_sio2,h,a,b)
    f1_si,  f2_si   = predict_batchSI(model_si, h,a,b)
    W_max_list.append(f1_w)
    W_ave_list.append(f2_w)

    SiO2_max_list.append(f1_sio2)
    SiO2_ave_list.append(f2_sio2)

    Si_max_list.append(f1_si)
    Si_ave_list.append(f2_si)

# 转成 numpy array，方便画图
W_max_list   = np.array(W_max_list)
W_ave_list   = np.array(W_ave_list)
SiO2_max_list = np.array(SiO2_max_list)
SiO2_ave_list = np.array(SiO2_ave_list)
Si_max_list   = np.array(Si_max_list)
Si_ave_list   = np.array(Si_ave_list)


iters = np.arange(1, len(X_path) + 1)

plt.figure(figsize=(7,5))

plt.plot(iters, W_max_list / w_max,   "-o", label="W (max)")
plt.plot(iters, SiO2_max_list / sio2_max,"-s", label="SiO$_2$ (max)")
plt.plot(iters, Si_max_list / si_max,  "-^", label="Si (max)")

plt.xlabel("Iteration")
plt.ylabel("Maximum stress (MPa)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,5))

plt.plot(iters, W_ave_list,   "-o", label="W (average)")
plt.plot(iters, SiO2_ave_list,"-s", label="SiO$_2$ (average)")
plt.plot(iters, Si_ave_list,  "-^", label="Si (average)")

plt.xlabel("Iteration")
plt.ylabel("Average stress (MPa)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,5))

plt.plot(W_path, W_max_list,   "-o", label="W (max)")
plt.plot(W_path, SiO2_max_list,"-s", label="SiO$_2$ (max)")
plt.plot(W_path, Si_max_list,  "-^", label="Si (max)")

plt.xlabel("Objective value w")
plt.ylabel("Maximum stress (MPa)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================
# 输出最优解的具体参数
# ===============================
best_params = X_path[best_id]

print("-" * 30)
print("Optimization Results:")
print(f"Best Iteration: {best_id}")
print(f"Minimum Objective w: {W_path[best_id]:.6f}")
print("-" * 30)
print(f"Optimal h: {best_params[0]:.4f}")
print(f"Optimal a: {best_params[1]:.4f}")
print(f"Optimal b: {best_params[2]:.4f}")
print("-" * 30)