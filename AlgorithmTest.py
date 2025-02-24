import math
from itertools import combinations
from collections import defaultdict
import networkx as nx
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

pr76_dict = {
    1: (3600, 2300), 2: (3100, 3300), 3: (4700, 5750), 4: (5400, 5750), 5: (5608, 7103), 6: (4493, 7102),
    7: (3600, 6950), 8: (3100, 7250), 9: (4700, 8450), 10: (5400, 8450), 11: (5610, 10053), 12: (4492, 10052),
    13: (3600, 10800), 14: (3100, 10950), 15: (4700, 11650), 16: (5400, 11650), 17: (6650, 10800), 18: (7300, 10950),
    19: (7300, 7250), 20: (6650, 6950), 21: (7300, 3300), 22: (6650, 2300), 23: (5400, 1600), 24: (8350, 2300),
    25: (7850, 3300), 26: (9450, 5750), 27: (10150, 5750), 28: (10358, 7103), 29: (9243, 7102), 30: (8350, 6950),
    31: (7850, 7250), 32: (9450, 8450), 33: (10150, 8450), 34: (10360, 10053), 35: (9242, 10052), 36: (8350, 10800),
    37: (7850, 10950), 38: (9450, 11650), 39: (10150, 11650), 40: (11400, 10800), 41: (12050, 10950), 42: (12050, 7250),
    43: (11400, 6950), 44: (12050, 3300), 45: (11400, 2300), 46: (10150, 1600), 47: (13100, 2300), 48: (12600, 3300),
    49: (14200, 5750), 50: (14900, 5750), 51: (15108, 7103), 52: (13993, 7102), 53: (13100, 6950), 54: (12600, 7250),
    55: (14200, 8450), 56: (14900, 8450), 57: (15110, 10053), 58: (13992, 10052), 59: (13100, 10800), 60: (12600, 10950),
    61: (14200, 11650), 62: (14900, 11650), 63: (16150, 10800), 64: (16800, 10950), 65: (16800, 7250), 66: (16150, 6950),
    67: (16800, 3300), 68: (16150, 2300), 69: (14900, 1600), 70: (19800, 800), 71: (19800, 10000), 72: (19800, 11900),
    73: (19800, 12200), 74: (200, 12200), 75: (200, 1100), 76: (200, 800)
}




def distance(point1_id, point2_id, data):
    """计算两点间的欧氏距禨"""
    return math.sqrt((data[point1_id][0] - data[point2_id][0]) ** 2 + (data[point1_id][1] - data[point2_id][1]) ** 2)

def total_distance(tour, data):
    """计算TSP解的总距离"""
    return sum(distance(tour[i], tour[i + 1], data) for i in range(len(tour) - 1)) + distance(tour[-1], tour[0], data)


def build_distance_matrix(data):
    """构建所有节点间的距离矩阵"""
    n = len(data)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = distance(i + 1, j + 1, data)
    return distance_matrix

def christofides(data):
    """基于克里斯托夫启发式算法求解TSP问题"""
    n = len(data)
    distance_matrix = build_distance_matrix(data)

    def minimum_spanning_tree(distance_matrix):
        """Prim算法生成最小生成树"""
        INF = float('inf')
        mst = [0]
        min_dist = [INF] * n
        pre_vertex = [0] * n
        while len(mst) < n:
            u = mst[-1]
            min_dist[u] = INF
            for v in range(n):
                if v not in mst and distance_matrix[u][v] < min_dist[v]:
                    min_dist[v] = distance_matrix[u][v]
                    pre_vertex[v] = u
            mst.append(min_dist.index(min(min_dist)))
        print(pre_vertex)
        return pre_vertex

    def find_odd_vertex(pre_vertex):
        """Find vertices with odd degrees in the minimum spanning tree"""
        degree_count = [0] * len(pre_vertex)

        for vertex in pre_vertex:
            if vertex != -1:
                degree_count[vertex] += 1
                degree_count[pre_vertex.index(vertex)] += 1

        odd_vertex = [i for i, degree in enumerate(degree_count) if degree % 2 != 0]
        print(len(odd_vertex), odd_vertex)
        return odd_vertex

    def minimum_weight_matching(distance_matrix, odd_vertex):
        """Minimum weight matching using networkx"""
        G = nx.Graph()

        # Add edges between odd degree vertices with weights
        for i in range(len(odd_vertex)):
            for j in range(i + 1, len(odd_vertex)):
                u, v = odd_vertex[i], odd_vertex[j]
                weight = distance_matrix[u][v]
                G.add_edge(u, v, weight=weight)

        # Find the minimum weight matching
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight='weight')

        # Convert the matching to the required format
        match = [-1] * len(distance_matrix)
        for u, v in matching:
            match[u] = v
            match[v] = u

        print(match)
        return match



    def find_euler_circuit(mst, min_weight_matching):
        """找到欧拉回路"""
        euler_circuit = []
        for i in range(n):
            if min_weight_matching[i] != -1:
                euler_circuit.append((min_weight_matching[i], i))
        for i in range(n):
            if mst[i] != -1:
                euler_circuit.append((mst[i], i))

        print(23232, euler_circuit)
        return euler_circuit

    def find_tsp_tour(euler_circuit):
        """生成TSP解"""
        def dfs(u):
            visited[u] = True
            for v in range(n):
                if not visited[v] and distance_matrix[u][v] != 0:
                    visited[v] = True
                    dfs(v)
            path.append(u)

        path = []
        visited = [False] * n
        dfs(euler_circuit[0][0])
        path.reverse()
        return path

    # Step 1: 生成最小生成树
    mst = minimum_spanning_tree(distance_matrix)
    # Step 2: 找到最小度数为奇数的节点
    odd_vertex = find_odd_vertex(mst)
    # Step 3: 最小权匹配
    min_weight_matching = minimum_weight_matching(distance_matrix, odd_vertex)
    # Step 4: 生成欧拉回路
    euler_circuit = find_euler_circuit(mst, min_weight_matching)
    # Step 5: 生成TSP解
    tsp_tour = find_tsp_tour(euler_circuit)

    for i in range(len(tsp_tour)):
        tsp_tour[i] += 1


    return tsp_tour,  total_distance(tsp_tour, data)


print(christofides(pr76_dict))
print('  ')



# -----------------------------------
# 1. 计算距离矩阵
# -----------------------------------
def calculate_distance(coord1, coord2):
    """计算两个坐标点之间的欧几里得距离"""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def build_distance_matrix(coords):
    """构建所有节点间的距离矩阵"""
    n = len(coords)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = calculate_distance(coords[i], coords[j])
    return distance_matrix


# -----------------------------------
# 2. Prim算法构建最小生成树（MST）
# -----------------------------------
def prim_mst(distance_matrix):
    """使用Prim算法构建最小生成树"""
    n = len(distance_matrix)
    visited = [False] * n
    parent = [-1] * n
    key = [float('inf')] * n
    key[0] = 0  # 从节点0开始

    for _ in range(n):
        # 找到未访问的最小key值节点
        u = min((k for k in range(n) if not visited[k]), key=lambda x: key[x])
        visited[u] = True
        for v in range(n):
            if not visited[v] and distance_matrix[u][v] < key[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u


    # 构建邻接表表示的MST
    mst = defaultdict(list)
    for v in range(1, n):
        u = parent[v]
        weight = distance_matrix[u][v]
        mst[u].append((v, weight))
        mst[v].append((u, weight))
    print(mst)
    return mst


# -----------------------------------
# 3. 找到奇数度数的节点
# -----------------------------------
def find_odd_degree_nodes(mst):
    """在MST中找到所有度数为奇数的节点"""
    odd = []
    for u in mst.keys():
        if len(mst[u]) % 2 != 0:
            odd.append(u)
    print(len(odd), odd)
    return odd


# -----------------------------------
# 4. 最小权重完美匹配（Blossom算法的简化实现）
# -----------------------------------
def minimum_weight_matching(odd_nodes, distance_matrix):
    """使用networkx库实现Blossom算法的最小权重完美匹配"""

    G = nx.Graph()
    for i in odd_nodes:
        for j in odd_nodes:
            if i != j and i < j:
                G.add_edge(i, j, weight=distance_matrix[i][j])  # 负权重

    matches = nx.min_weight_matching(G)
    print(matches)
    return list(matches)


# -----------------------------------
# 5. 生成欧拉回路和哈密顿路径
# -----------------------------------
def find_eulerian_tour(multigraph):
    """使用Hierholzer算法找到欧拉回路"""
    stack = []
    circuit = []
    current_node = next(iter(multigraph))  # 起始节点

    while True:
        if multigraph[current_node]:
            stack.append(current_node)
            next_node, _ = multigraph[current_node].pop()
            current_node = next_node
        else:
            circuit.append(current_node)
            if not stack:
                break
            current_node = stack.pop()
    return circuit[::-1]  # 反转得到正确顺序


def shortcut_eulerian_tour(euler_tour):
    """通过短路生成哈密顿路径"""
    visited = set()
    tsp_path = []
    for node in euler_tour:
        if node not in visited:
            visited.add(node)
            tsp_path.append(node)
    # 回到起点形成环路
    tsp_path.append(tsp_path[0])
    return tsp_path


# -----------------------------------
# 主函数：Christofides算法
# -----------------------------------
def christofides_tsp(coords):
    # 步骤1: 构建距离矩阵
    distance_matrix = build_distance_matrix(coords)

    # 步骤2: 构建最小生成树
    mst = prim_mst(distance_matrix)

    # 步骤3: 找到奇数度节点
    odd_nodes = find_odd_degree_nodes(mst)

    # 步骤4: 最小权重完美匹配
    matches = minimum_weight_matching(odd_nodes, distance_matrix)

    # 构建欧拉图（合并MST和匹配边）
    multigraph = defaultdict(list)
    for u in mst:
        for v, w in mst[u]:
            multigraph[u].append((v, w))
    for u, v in matches:
        multigraph[u].append((v, distance_matrix[u][v]))
        multigraph[v].append((u, distance_matrix[u][v]))

    # 步骤5: 生成欧拉回路并短路为TSP路径
    euler_tour = find_eulerian_tour(multigraph)
    tsp_path = shortcut_eulerian_tour(euler_tour)

    # 计算总距离
    total_distance = 0
    for i in range(len(tsp_path) - 1):
        total_distance += distance_matrix[tsp_path[i]][tsp_path[i + 1]]

    return tsp_path, total_distance

def random_clip(path, num, max_point_num):
    """将该路径按原顺序随机剪成num段，每段的点数不超过max_point_num"""
    size = len(path) - 1
    clip_tour = []
    if size > num * max_point_num:
        print("路径过长，无法剪裁")
        return
    if max_point_num >= size - num + 1:
        indices = random.sample(range(1, size), num - 1)
        for i in range(len(indices) - 1):
            clip_tour.append(path[indices[i]:indices[i + 1]])
        return clip_tour
    st = 0
    for i in range(num - 1):
        ed = random.randint(max(st + 1, size - (num - 1 - i) * max_point_num), min(st + max_point_num, size))
        clip_tour.append(path[st:ed])
        st = ed
    return clip_tour





def pmx_crossover(parent1, parent2):
    size = len(parent1)
    # 随机选择两个交叉点
    cx1 = random.randint(0, size - 1)
    cx2 = random.randint(cx1 + 1, size)

    # 初始化子代
    child1 = parent1.copy()
    child2 = parent2.copy()

    # 建立映射关系
    mapping1 = {}
    mapping2 = {}
    for i in range(cx1, cx2):
        val1 = parent1[i]
        val2 = parent2[i]
        mapping1[val2] = val1  # parent2的值映射到parent1
        mapping2[val1] = val2  # parent1的值映射到parent2

    # 填充子代
    for i in range(size):
        if i < cx1 or i >= cx2:
            # 处理子代1
            while child1[i] in mapping1:
                child1[i] = mapping1[child1[i]]
            # 处理子代2
            while child2[i] in mapping2:
                child2[i] = mapping2[child2[i]]

    return child1, child2

def swap_mutation(tour1, tour2, idx1, idx2):
    tour1[idx1], tour2[idx2] = tour2[idx2], tour1[idx1]
    return tour1, tour2



# -----------------------------------
# 数据准备与执行
# -----------------------------------
if __name__ == "__main__":

    N = 50 #种群大小
    p_r = 0.95 #交叉概率
    iter = 30  #迭代次数
    p_m = 0.1 / 76 #变异概率文章里根本没说

    # 数据初始化部分（Christofides）
    coords = [pr76_dict[i + 1] for i in range(len(pr76_dict))]  # ID 1~76 对应索引0~75
    tsp_path_indices, total_distance = christofides_tsp(coords)
    tsp_path = [idx + 1 for idx in tsp_path_indices]
    print("TSP路径（节点ID顺序）:", tsp_path)
    print("总距离:", total_distance)

    # 数据初始化部分（LKH）



