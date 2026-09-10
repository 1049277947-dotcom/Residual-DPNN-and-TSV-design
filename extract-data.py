import pandas as pd
import numpy as np
import random

# 设置随机种子为None，确保每次运行都不同
# 或者可以使用当前时间作为种子：random_seed = int(time.time())
random_seed = None

# 读取数据
df = pd.read_excel('SI_Train_Pool_80.xlsx')

# 定义抽样比例
sampling_ratios = [0.1]

# 对每个x值分别进行抽样
sampled_dfs = {}
for ratio in sampling_ratios:
    sampled_data = []
    
    # 按x值分组
    for x_value in df['x'].unique():
        # 获取当前x值的所有数据
        x_data = df[df['x'] == x_value]
        
        # 随机抽样指定比例，不重复抽样，每次使用不同的随机状态
        # 使用不同的随机种子确保每次运行结果不同
        current_seed = random.randint(1, 10000) if random_seed is None else random_seed + int(ratio*100)
        sampled_x = x_data.sample(frac=ratio, random_state=current_seed, replace=False)
        sampled_data.append(sampled_x)
    
    # 合并所有抽样数据
    sampled_df = pd.concat(sampled_data, ignore_index=True)
    sampled_dfs[f'{int(ratio*100)}%'] = sampled_df
    
    # 打印抽样信息
    print(f"{int(ratio*100)}% 数据抽样完成: 原始数据 {len(df)} 行, 抽样后 {len(sampled_df)} 行")

# 保存到新的Excel文件，每个比例一个sheet
with pd.ExcelWriter('SI_Train_Pool_80_104.xlsx') as writer:
    for name, sampled_df in sampled_dfs.items():
        sampled_df.to_excel(writer, sheet_name=name, index=False)

print("所有抽样数据已保存到 SI_sampled_random.xlsx")