import csv
import pandas as pd
import numpy as np
import itertools

from networkx.algorithms.traversal import dfs_successors


# # 生成 0~164 的所有两两组合
# numbers = range(167)  # 0~164
#
# combinations = []  # 生成所有两两组合
# for i in range(167):
#     for j in range(167):
#         if i != j: combinations.append((i, j))

# 写入 CSV 文件
# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入表头
#     writer.writerow(['O', 'D', 'Ton'])
#     # 写入数据
#     for combo in combinations:
#         writer.writerow([combo[0], combo[1], 0])  # Ton 列填充默认值 0


#
# adj_list = []
# adj_dict = {}
#
# with open('PY_net.csv', mode='r') as file:
#     reader = csv.DictReader(file, delimiter=',')
#     print(reader)
#
#     for row in reader:
#         print(row)
#         adj_list.append([row['init_node'], row['term_node']])
#         if row['init_node'] not in adj_dict.keys():
#             adj_dict[row['init_node']] = []
#         adj_dict[row['init_node']].append(row['term_node'])
#
# print(adj_list)
# print(adj_dict)
#
# rows = []
# with open('output.csv', mode='r') as file:
#     reader = csv.DictReader(file, delimiter=',')
#     for row in reader:
#         if [row['O'], row['D']] in adj_list:
#             row['Ton'] = 0
#             row['adj'] = 1
#         else:
#             flag = 0
#             for neighbor in adj_dict[row['O']]:
#                 if row['D'] in adj_dict[neighbor]:
#                     row['Ton'] = 0
#                     row['adj'] = 1 if row['adj'] == 1 else 2
#                     flag = 1
#                     break
#             if flag == 0:
#                 row['Ton'] = 1
#                 row['adj'] = 0
#
#         rows.append(row)
#
# # 将修改后的数据写回 CSV 文件
# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=rows[0].keys())
#     writer.writeheader()  # 写入表头
#     writer.writerows(rows)  # 写入数据
#
# print("CSV 文件已更新！")
#


rows = []
cnt = 4
with open('output.csv', mode='r') as file:
    reader = csv.DictReader(file, delimiter=',')
    for row in reader:
        if row['adj'] == '0':
            if row['O'] != '165' and row['O'] != '166' and row['D'] != '165' and row['D'] != '166':
                cnt += 1
            if cnt % 17 == 0:
                row['Ton'] = 100
                rows.append(row)



 

# 将修改后的数据写回 CSV 文件
with open('PY_od4.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()  # 写入表头
    writer.writerows(rows)  # 写入数据

print("CSV 文件已更新！")