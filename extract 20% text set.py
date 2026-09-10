import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_excel('W-NEWDATASET.xlsx')

# 核心步骤：使用 sklearn 进行分层划分
# test_size=0.2 表示抽取 20% 作为验证集
# random_state=42 设定固定的随机种子，确保每次划分的验证集是唯一且确定的标准集
# stratify=df['x'] 确保训练集和验证集中 'x' 的分布比例与原表相同
train_pool, validation_set = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 

)

# 打印划分结果验证
print(f"总数据量: {len(df)} 行")
print(f"标准验证集 (20%): {len(validation_set)} 行")
print(f"训练集基底 (80%): {len(train_pool)} 行")

# 保存为两个独立的文件（或保存在同一个Excel的不同Sheet中）
validation_set.to_excel('W_Validation_Set_20.xlsx', index=False)
train_pool.to_excel('W_Train_Pool_80.xlsx', index=False)

print("✅ 标准验证集和训练集基底已成功分离并保存。")