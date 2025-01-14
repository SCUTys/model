##原论文使用oliver30数据集来实现的,数据集写距离函数用的伪欧式距离,但论文里是欧氏距离

import math
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor


def pseudo_euclidean_distance(x1, y1, x2, y2):
    return math.ceil(math.sqrt(((x1 - x2) ** 2  + (y1 - y2) ** 2)/ 1.0))


def heuristic_greedy(adj_matrix):
    greedy_path = []
    greedy_path.append(random.randint(1, 30))
    while len(greedy_path) < 30:
        index = greedy_path[-1] - 1
        min_distance = float('inf')
        next_index = -1

        for i in range(30):
            if i + 1 not in greedy_path:
                if adj_matrix[index][i] < min_distance:
                    min_distance = adj_matrix[index][i]
                    next_index = i

        greedy_path.append(next_index + 1)

    greedy_path.append(greedy_path[0])
    greedy_sum = 0
    for i in range(30):
        greedy_sum += adj_matrix[greedy_path[i] - 1][greedy_path[i + 1] - 1]

    return greedy_sum, greedy_path


def calculate_path_distance(path, adj_matrix):
    path_distance = 0
    for i in range(30):
        # print(path[i], path[i + 1], adj_matrix[path[i] - 1][path[i + 1] - 1])
        path_distance += adj_matrix[path[i] - 1][path[i + 1] - 1]
    return path_distance


data_dict = {5: [2, 99],
             10: [4, 50],
             6: [7, 64],
             11: [13, 40],
             12: [18, 40],
             9: [18, 54],
             8: [22, 60],
             13: [24, 42],
             14: [25, 38],
             7: [25, 62],
             3: [37, 84],
             16: [41, 26],
             4: [41, 94],
             15: [44, 35],
             17: [45, 21],
             2: [54, 62],
             1: [54, 67],
             18: [58, 35],
             30: [58, 69],
             19: [62, 32],
             24: [64, 60],
             25: [68, 58],
             23: [71, 44],
             29: [71, 71],
             28: [74, 78],
             20: [82, 7],
             22: [83, 46],
             26: [83, 69],
             27: [87, 76],
             21: [91, 38]}

beta = 2
q0 = 0.9
alpha = 0.1
rho = 0.1
m = 10
iter = 2500

adj_matrix = np.zeros((30, 30))
for i in range(30):
    for j in range(i, 30):
        if i == j:
            adj_matrix[i][j] = float('inf')
        else:
            adj_matrix[i][j] = adj_matrix[j][i] = pseudo_euclidean_distance(data_dict[i + 1][0], data_dict[i + 1][1],
                                                                            data_dict[j + 1][0], data_dict[j + 1][1])

greedy_distance, greedy_path = heuristic_greedy(adj_matrix)
tao_0 = 1.0 / (30 * greedy_distance)

tao = [[tao_0 for i in range(30)] for j in range(30)]
neta = [[1.0 / adj_matrix[i][j] for i in range(30)] for j in range(30)]

sol_distance = greedy_distance
sol_path = greedy_path

# print(adj_matrix)
print(greedy_distance)
print(greedy_path)

for it in range(iter):
    ant_start = random.sample(range(1, 31), m)
    ant_path = [[st] for st in ant_start]
    i = 1
    while i < 30:
        for ant in range(m):
            q = random.random()
            if q <= q0:
                max_prob = -1
                max_index = -1
                for j in range(30):
                    if j + 1 not in ant_path[ant]:
                        prob = (tao[ant_path[ant][-1] - 1][j]) * (neta[ant_path[ant][-1] - 1][j] ** beta)
                        if prob > max_prob:
                            max_prob = prob
                            max_index = j
                ant_path[ant].append(max_index + 1)
            else:
                select_index = [i for i in range(30) if i + 1 not in ant_path[ant]]
                prob = [tao[ant_path[ant][-1] - 1][j] * neta[ant_path[ant][-1] - 1][j] ** beta for j in select_index]
                prob = [p / sum(prob) for p in prob]
                next_index = np.random.choice(select_index, 1, p=prob)[0]
                ant_path[ant].append(next_index + 1)

        for ant in range(m):
            tao[ant_path[ant][-2] - 1][ant_path[ant][-1] - 1] = (1 - rho) * tao[ant_path[ant][-2] - 1][
                ant_path[ant][-1] - 1] + rho * tao_0

        i += 1

    for ant in range(m):
        ant_path[ant].append(ant_path[ant][0])

    # 计算路径长度
    for ant in range(m):
        # print(ant)
        # print(ant_path[ant])
        path_distance = calculate_path_distance(ant_path[ant], adj_matrix)
        if path_distance < sol_distance:
            sol_distance = path_distance
            sol_path = ant_path[ant]

    # 信息素全局更新
    for i in range(30):
        tao[sol_path[i] - 1][sol_path[i + 1] - 1] = (1 - alpha) * tao[sol_path[i] - 1][sol_path[i + 1] - 1] + alpha / sol_distance


    print(f"iter {it + 1}: {sol_distance}")
    print(sol_path)

