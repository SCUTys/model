import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 设置文件路径
root = Path(os.path.abspath('.')).parent / "pythonProject" / "data"
netfile = root / 'AT' /'Austin_net.tntp'

# netfile = r"D:\PyCharm 2024\PycharmProject\pythonProject\data\SF\SiouxFalls_net.tntp"
# 读取文件
net = pd.read_csv(netfile, skiprows=8, sep="\t")

# 处理列名
net.columns = [s.strip().lower() for s in net.columns]

# 删除不需要的列
net.drop(['~', ';'], axis=1, inplace=True)

output_file =  root / 'AT' / "AT_net_processed.csv"
net.to_csv(output_file, index=False)

# 打印前几行数据以验证
print(net.head())