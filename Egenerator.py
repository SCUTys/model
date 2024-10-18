import pandas as pd

# 读取CSV文件
df = pd.read_csv('data_3.csv')

# 定义需要操作的行
columns_to_modify = [7, 8, 9]

# 将指定列转换为浮点数类型，避免字符串混入
for col in columns_to_modify:
    df[col] = df[col].astype(float)

# 对指定列小于1.6的值增加0.4
for col in columns_to_modify:
    df.loc[df[col] < 1.6, col]

# 保存修改后的CSV文件
df.to_csv('modified_data_3.csv', index=False)