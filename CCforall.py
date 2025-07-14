import operator
import numpy as np
import simuplus
import math
import random
import heapq
import multiprocessing as mp
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import pandapower as pp
import copy
import time
from numba.cpython.randomimpl import permutation_impl
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from collections import defaultdict
from functools import lru_cache
import concurrent.futures

def process_od_length(real_path_results):
    od_length = {}
    for od, result in real_path_results.items():
        if od[0] == od[1]: continue
        sum = div = 0
        for i in range(len(result) - 1):
            sum += 1 / math.exp(result[i][1] + result[i][2])
            div += 1
        if div > 0:
            sum /= div
        else:
            sum = 1 / math.exp(result[0][1] + result[0][2])
        od_length[od] = sum
        od_length[(od[0], od[0])] = 0
        od_length[(od[1], od[1])] = 0

    od_wait = {}
    for od, result in real_path_results.items():
        sum = div = 0
        for i in range(len(result) - 1):
            sum += 1 / math.exp(result[0][2])
            div += 1
        if div > 0:
            sum /= div
        else:
            sum = 1 / math.exp(result[0][2])
        od_wait[od] = sum
    return od_length, od_wait


def process_cs(OD_ratio, cs, num_cs, od_length, anxiety = 1):
    cs_for_choice = {}
    for od in OD_ratio.keys():
        o, d = od
        if anxiety == 1:
            cs.sort(key=lambda x: od_length[(o, x)] + od_length[(x, d)], inverse = True)
        else:
            cs.sort(key=lambda x: od_length[(x, d)], inverse = True)
        cs_for_choice[od] = cs[:num_cs]
    return cs_for_choice


def calculate_hypervolume(fronts, reference_point):
    """
    计算二维前沿解集的超体积(HV)指标

    参数:
        fronts: 二维数组，每行表示一个解的坐标 [x, y]
        reference_point: 参考点坐标 [ref_x, ref_y]，应被前沿解集所有点支配

    返回:
        float: HV值
    """
    if not fronts:
        return 0.0

    # 确保前沿点都支配参考点（对于最小化问题，前沿点应小于参考点）
    valid_points = []
    for point in fronts:
        if all(point[i] <= reference_point[i] for i in range(len(point))):
            valid_points.append(point)

    if not valid_points:
        return 0.0

    # 对于二维情况，按第一维升序排序
    sorted_points = sorted(valid_points, key=lambda x: x[0])

    # 计算HV (面积)
    hv = 0.0
    prev_y = reference_point[1]

    for i in range(len(sorted_points)):
        point = sorted_points[i]

        # 计算当前点贡献的矩形面积
        if i == len(sorted_points) - 1:
            width = reference_point[0] - point[0]
        else:
            width = sorted_points[i + 1][0] - point[0]

        height = prev_y - point[1]
        hv += width * height

        # 更新prev_y为当前点的y值
        prev_y = point[1]

    return hv


def find_best_contributor(scores, F1_cnt):
    max_f1 = max(score[0] for score in scores) * 1.1  # 增加10%余量
    max_f2 = max(score[1] for score in scores) * 1.1
    reference_point = [max_f1, max_f2]

    # 计算完整前沿的超体积
    first_front_scores = scores[:F1_cnt]
    total = calculate_hypervolume(first_front_scores, reference_point)

    # 计算每个点的贡献
    contributions = []
    for i in range(F1_cnt):
        # 移除当前点后计算超体积
        remaining_scores = first_front_scores[:i] + first_front_scores[i + 1:]
        hv_without_i = calculate_hypervolume(remaining_scores, reference_point)

        # 贡献 = 完整超体积 - 没有该点的超体积
        contribution = total - hv_without_i
        contributions.append(contribution)

    # 找到贡献最大的点的索引
    best_idx = contributions.index(max(contributions))
    return best_idx




def generate_random_1(l):
    numbers = [random.random() for _ in range(l)]
    total = sum(numbers)
    if total > 0:
        normalized_numbers = [n / total for n in numbers]
    else:
        # 处理极端情况，所有随机数都为0
        normalized_numbers = [1.0 / l for _ in range(l)]
    return normalized_numbers


def generate_random_individual(num_cs, num_path):
    """
    生成一个随机个体，表示为num_cs个子向量，每个子向量长度为num_path
    每个子向量中的元素范围为0~1，且总和为1

    返回:
        list: 包含num_cs个子向量的列表，每个子向量长度为num_path
    """
    individual = []
    weight = generate_random_1(num_cs)
    for i in range(num_cs):
        sub_vector = generate_random_1(num_path)
        subs = [x * weight[i] for x in sub_vector]
        for sub in subs:
            individual.append(sub)

    return individual

def initialize_ccMODE(num_population, num_cs, num_path):
    """
    初始化ccMODE算法的参数和数据结构
    """
    # 初始化种群
    population = []
    for _ in range(num_population):
        individual = generate_random_individual(num_cs, num_path)
        population.append(individual)

    return population


def f1(individual, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice,
       path_for_choice, anxiety_path_for_choice, k, num_cs):
    """
    计算目标函数f1：所有选择路径的总时间之和

    参数:
        individual: 个体，表示为权重向量
        lmp_dict: 节点边际价格字典
        OD_ratio: 普通OD对及其比例
        anxiety_OD_ratio: 焦虑OD对及其比例
        cs: 充电站列表
        cs_bus: 充电站对应的电网节点
        cs_for_choice: 每个普通OD对可选的充电站
        anxiety_cs_for_choice: 每个焦虑OD对可选的充电站
        path_for_choice: 每个普通OD-CS组合的路径
        anxiety_path_for_choice: 每个焦虑OD-CS组合的路径
        k: 每个OD-CS组合的路径数量
        num_cs: 每个OD对选择的充电站数量

    返回:
        float: 总时间值
    """
    f1_value = 0
    od_len_f1 = len(OD_ratio.keys())
    cs_choice_cnt = num_cs
    path_choice_cnt = k
    part = individual[:od_len_f1 * cs_choice_cnt * path_choice_cnt]
    anxiety_part = individual[od_len_f1 * cs_choice_cnt * path_choice_cnt:]

    # 计算普通OD对的总时间
    for i in range(od_len_f1):
        od = list(OD_ratio.keys())[i]
        o, d = od
        for j in range(cs_choice_cnt):
            cs_id = cs_for_choice[od][j]
            for kk in range(path_choice_cnt):
                weight = part[i * cs_choice_cnt * path_choice_cnt + j * path_choice_cnt + kk]
                # 将权重乘以OD比例和路径总时间
                if (o, cs_id, d) in path_for_choice and path_for_choice[(o, cs_id, d)]:
                    f1_value += weight * OD_ratio[od] * path_for_choice[(o, cs_id, d)][kk]['total_time']

    # 计算焦虑OD对的总时间
    anxiety_od_len = len(anxiety_OD_ratio.keys())
    for i in range(anxiety_od_len):
        od = list(anxiety_OD_ratio.keys())[i]
        o, d = od
        for j in range(cs_choice_cnt):
            cs_id = anxiety_cs_for_choice[od][j]
            for kk in range(path_choice_cnt):
                weight = anxiety_part[i * cs_choice_cnt * path_choice_cnt + j * path_choice_cnt + kk]
                # 将权重乘以OD比例和路径总时间
                if (o, d, cs_id) in anxiety_path_for_choice and anxiety_path_for_choice[(o, d, cs_id)]:
                    f1_value += weight * anxiety_OD_ratio[od] * anxiety_path_for_choice[(o, d, cs_id)][kk]['total_time']

    return f1_value



def f2(individual, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, path_for_choice, anxiety_path_for_choice, k, num_cs):
    lmp = {}
    bus_node = {}
    for id in cs_bus:
        lmp[id] = lmp_dict[id]
    for i in range(len(cs)):
        bus_node[cs[i]] = cs_bus[i]
    sum_value = sum(lmp.values())
    for key in lmp:
        lmp[key] /= sum_value

    f2_value = 0
    od_len_f2 = len(OD_ratio.keys())
    cs_choice_cnt = num_cs
    path_choice_cnt = k
    part = individual[:od_len_f2 * cs_choice_cnt * path_choice_cnt]
    anxiety_part = individual[od_len_f2 * cs_choice_cnt * path_choice_cnt:]

    for i in range(od_len_f2):
        od = list(OD_ratio.keys())[i]
        o, d = od
        for j in range(cs_choice_cnt):
            cs_id = cs_for_choice[od][j]
            for k in range(path_choice_cnt):
                f2_value += (part[i * cs_choice_cnt * path_choice_cnt + j * path_choice_cnt + k] * OD_ratio[od] * lmp[bus_node[cs_id]]
                             * (54 - 4 + path_for_choice[(o, cs_id, d)][k]['first_part_travel_time'] * 0.15
                                + path_for_choice[(o, cs_id, d)][k]['first_part_wait_time'] * 0.05))

    for i in range(len(anxiety_OD_ratio.keys())):
        od = list(anxiety_OD_ratio.keys())[i]
        o, d = od
        for j in range(cs_choice_cnt):
            cs_id = anxiety_cs_for_choice[od][j]
            for k in range(path_choice_cnt):
                f2_value += (anxiety_part[i * cs_choice_cnt * path_choice_cnt + j * path_choice_cnt + k] * anxiety_OD_ratio[od] * lmp[bus_node[cs_id]]
                             * (54 - 6 + 0.15 * anxiety_path_for_choice[(o, d, cs_id)][k]['travel_time']
                                + 0.05 * anxiety_path_for_choice[(o, d, cs_id)][k]['wait_time']))

    return f2_value


def find_k_shortest_paths_via_stop(graph, node_weights, start, stop, end, k=3):
    """
    寻找从起点经过指定中间点到终点的k条最短路径，考虑行驶时长和节点等待时长

    参数:
        center: 中心对象
        graph: defaultdict，表示道路网络，格式为 {(origin, destination): travel_time}
        node_weights: defaultdict，节点等待时长，格式为 {(prev_node, node, next_node): wait_time}
        start: 起点ID
        stop: 经停点ID
        end: 终点ID
        k: 返回的路径数量

    返回:
        list: 包含k条最短路径的列表，每条路径是一个字典，包含'path'、'travel_time'和'wait_time'
    """
    # 构建邻接表
    adj_list = defaultdict(list)
    for (src, dst), weight in graph.items():
        adj_list[src].append((dst, weight))

    # 辅助函数：使用Dijkstra算法寻找单源最短路径，并考虑节点等待时间
    def dijkstra(start_node, end_node, adj):
        pq = [(0, 0, start_node, [start_node], None)]  # (总时间, 等待时间, 当前节点, 路径, 前一节点)
        visited = set()
        paths = []

        while pq and len(paths) < k:
            (total_time, wait_time, current, path, prev_node) = heapq.heappop(pq)

            # 使用(当前节点, 路径长度)作为访问标记，允许同一节点在不同长度的路径中被访问
            visit_key = (current, len(path))
            if visit_key in visited:
                continue

            if current == end_node:
                paths.append({
                    'path': path,
                    'travel_time': total_time - wait_time,
                    'wait_time': wait_time,
                    'total_time': total_time
                })
                continue

            visited.add(visit_key)

            for neighbor, travel_time in adj[current]:
                if (neighbor, len(path) + 1) not in visited:
                    # 计算等待时间（如果有前一节点和下一节点）
                    new_wait_time = wait_time
                    if prev_node is not None:
                        wait_key = (prev_node, current, neighbor)
                        if wait_key in node_weights:
                            new_wait_time += node_weights[wait_key]

                    # 总时间 = 已有总时间 + 行驶时间 + 可能的等待时间
                    new_total_time = total_time + travel_time

                    heapq.heappush(pq, (
                        new_total_time,
                        new_wait_time,
                        neighbor,
                        path + [neighbor],
                        current
                    ))

        return paths

    # 1. 寻找从起点到经停点的k条最短路径
    paths_to_stop = dijkstra(start, stop, adj_list)

    # 2. 寻找从经停点到终点的k条最短路径
    paths_from_stop = dijkstra(stop, end, adj_list)

    # 3. 组合路径并计算总时间
    combined_paths = []

    for path1 in paths_to_stop:
        for path2 in paths_from_stop:
            # 合并路径（去掉重复的经停点）
            full_path = path1['path'] + path2['path'][1:]

            # 计算总行驶时间
            travel_time = path1['travel_time'] + path2['travel_time']

            # 计算总等待时间
            wait_time = path1['wait_time'] + path2['wait_time']

            # 还要考虑经停点的等待时间
            if len(path1['path']) >= 2 and len(path2['path']) >= 2:
                last_node_in_path1 = path1['path'][-2]  # 经停点前的节点
                first_node_in_path2 = path2['path'][1]  # 经停点后的节点
                wait_key = (last_node_in_path1, stop, first_node_in_path2)
                if wait_key in node_weights:
                    wait_time += node_weights[wait_key]

            # 总时间 = 行驶时间 + 等待时间
            total_time = travel_time + wait_time

            combined_paths.append({
                'path': full_path,
                'travel_time': travel_time,
                'wait_time': wait_time,
                'total_time': total_time,
                'first_part_travel_time': path1['travel_time'],
                'second_part_travel_time': path2['travel_time'],
                'first_part_wait_time': path1['wait_time'],
                'second_part_wait_time': path2['wait_time'],
            })

    # 4. 按总时间排序并返回k条最短路径
    combined_paths.sort(key=lambda x: x['total_time'])
    return combined_paths[:k]


def process_path_results(edge_weights, node_weights, OD_ratio, cs, k, num_cs, anxiety=1):
    """
    处理路径结果并返回每个OD对的最优路径选择和效用最高的充电站

    参数:
        edge_weights: 边的权重(行驶时间)
        node_weights: 节点权重(等待时间)
        OD_ratio: OD对及其比例
        cs: 充电站列表
        k: 每个OD-CS组合需要的最短路径数量
        anxiety: 是否为焦虑路径(1表示正常路径，0表示焦虑路径)

    返回:
        tuple: (path_for_choice, best_cs_for_od)
            - path_for_choice: 每个(o,cs,d)组合的k条最短路径
            - best_cs_for_od: 每个OD对效用最高的四个充电站ID
    """
    path_for_choice = {}
    # 计算每个OD对到每个充电站的效用
    od_cs_utility = {}

    for od in OD_ratio.keys():
        o, d = od
        od_cs_utility[od] = []

        # 计算每个充电站的效用
        for cs_id in cs:
            if anxiety == 1:
                # 正常路径: o->cs->d 的效用 (效用越低越好，所以使用总时间的负值)
                paths = find_k_shortest_paths_via_stop(edge_weights, node_weights, o, cs_id, d, k)
                # 取第一条路径(最短路径)计算效用
                if paths:
                    utility = 0
                    for cnt in range(k):
                        utility += exp(-0.1*(paths[cnt]['total_time'])) / k
                else:
                    print(f"Warning: No paths found for OD {od} via CS {cs_id}. Using negative infinity as utility.{anxiety}")
                    utility = -1
                path_for_choice[(o, cs_id, d)] = paths
            else:
                paths = find_k_shortest_paths_via_stop(edge_weights, node_weights, o, d, cs_id, k)
                if paths:
                    utility = 0
                    for cnt in range(k):
                        utility += exp(-0.1 * (paths[cnt]['total_time'])) / k
                else:
                    print(
                        f"Warning: No paths found for OD {od} via CS {cs_id}. Using negative infinity as utility.{anxiety}")
                    utility = -1
                path_for_choice[(o, d, cs_id)] = paths

            od_cs_utility[od].append((cs_id, utility))

    # 为每个OD对选择效用最高的四个充电站
    best_cs_for_od = {}
    for od in OD_ratio.keys():
        # 按效用降序排序
        od_cs_utility[od].sort(key=lambda x: x[1], reverse=True)
        # 选择前四个(或更少，如果充电站总数少于4)
        num_best = min(num_cs, len(od_cs_utility[od]))
        best_cs_for_od[od] = [cs_pair[0] for cs_pair in od_cs_utility[od][:num_best]]

    return path_for_choice, best_cs_for_od


def index_to_od(index, OD_ratio, anxiety_OD_ratio=None):
    l1 = len(OD_ratio.keys())
    if anxiety_OD_ratio:
        l2 = len(anxiety_OD_ratio.keys())
    else:
        l2 = 0

    if index < l1:
        od = list(OD_ratio.keys())[index]
    else:
        od = list(anxiety_OD_ratio.keys())[index - l1] if anxiety_OD_ratio else None
    return od


def select_random_indices(n, k, p):
    if k >= n - 1:
        # 如果要选择的数量大于等于可选范围，则返回所有可选数字
        return [i for i in range(n) if i != p]

    available = [i for i in range(n) if i != p]

    # 随机选择k个不同的索引
    return random.sample(available, k)


def rand_1(index, population, beta=0.4):
    ind = population[index]
    indices = select_random_indices(len(population), 2, index)
    # 这里需要进行元素级的计算
    return [ind[i] + beta * (population[indices[0]][i] - population[indices[1]][i])
            for i in range(len(ind))]


def rand_2(index, population, beta1=0.4, beta2=0.4):
    ind = population[index]
    indices = select_random_indices(len(ind), 4, index)
    # 实现元素级的计算
    return [ind[i] + beta1 * (population[indices[0]][i] - population[indices[1]][i])
            + beta2 * (population[indices[2]][i] - population[indices[3]][i])
            for i in range(len(ind))]


def simple_crossover(ind, mu, num_gene, mu_rate=0.5):
    """
    简单交叉操作，随机选择一个位置进行交叉
    """
    i = int(len(ind) / num_gene)
    off = []
    for j in range(i):
        r = random.random()
        if r < mu_rate:
            off.append(mu[j * num_gene:(j + 1) * num_gene])
        else:
            off.append(ind[j * num_gene:(j + 1) * num_gene])
    if off == ind:
        sum_mu = sum(mu)
        return [mu_gene / sum_mu for mu_gene in mu]
    else:
        sum_off = sum(off)
        return [off_gene / sum_off for off_gene in off]


def fast_non_dominated_sort_and_crowding(population, scores, population_size):
    """
    对种群进行快速非支配排序和拥挤度排序，选择前population_size个解

    参数:
        population: 种群，每个元素是一个个体
        scores: 每个个体对应的目标函数值列表，格式为 [[f1_1, f2_1], [f1_2, f2_2], ...]
        population_size: 返回的种群大小

    返回:
        tuple: (新种群, 前沿数量, 前沿解集, 前沿解集的指标对)
    """
    # 合并种群和得分为(个体, 得分)对
    combined = list(zip(population, scores))

    # 计算支配关系
    domination_counts = [0] * len(combined)  # 支配当前个体的其他个体数量
    dominated_sets = [[] for _ in range(len(combined))]  # 被当前个体支配的个体列表

    # 快速非支配排序
    for i in range(len(combined)):
        for j in range(len(combined)):
            if i == j:
                continue

            # 检查i是否支配j
            if (combined[i][1][0] <= combined[j][1][0] and combined[i][1][1] <= combined[j][1][1]) and \
                    (combined[i][1][0] < combined[j][1][0] or combined[i][1][1] < combined[j][1][1]):
                dominated_sets[i].append(j)
            # 检查j是否支配i
            elif (combined[j][1][0] <= combined[i][1][0] and combined[j][1][1] <= combined[i][1][1]) and \
                    (combined[j][1][0] < combined[i][1][0] or combined[j][1][1] < combined[i][1][1]):
                domination_counts[i] += 1

    # 按前沿对个体进行分组
    fronts = []
    current_front = []

    # 找到第一个前沿
    for i in range(len(combined)):
        if domination_counts[i] == 0:
            current_front.append(i)

    fronts.append(current_front)

    # 找到其余前沿
    i = 0
    while fronts[i]:
        next_front = []
        for j in fronts[i]:
            for k in dominated_sets[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    next_front.append(k)
        i += 1
        if next_front:
            fronts.append(next_front)

    # 计算拥挤度
    def calculate_crowding_distance(front_indices):
        if len(front_indices) <= 2:
            return {idx: float('inf') for idx in front_indices}

        distances = {idx: 0 for idx in front_indices}

        # 对每个目标函数排序并计算拥挤度
        for obj in range(2):  # 假设有2个目标函数
            sorted_front = sorted(front_indices, key=lambda idx: combined[idx][1][obj])

            # 设置边界点的拥挤度为无穷大
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # 计算中间点的拥挤度
            obj_range = combined[sorted_front[-1]][1][obj] - combined[sorted_front[0]][1][obj]
            if obj_range == 0:
                continue  # 避免除以零

            for i in range(1, len(sorted_front) - 1):
                distances[sorted_front[i]] += (combined[sorted_front[i + 1]][1][obj] -
                                               combined[sorted_front[i - 1]][1][obj]) / obj_range

        return distances

    # 为每个前沿计算拥挤度
    crowding_distances = {}
    for front in fronts:
        distances = calculate_crowding_distance(front)
        crowding_distances.update(distances)

    # 按前沿和拥挤度选择新种群
    selected_indices = []
    i = 0
    while len(selected_indices) + len(fronts[i]) <= population_size:
        selected_indices.extend(fronts[i])
        i += 1
        if i >= len(fronts):
            break

    # 如果还需要更多个体，则根据拥挤度选择
    if len(selected_indices) < population_size and i < len(fronts):
        remaining = sorted(fronts[i], key=lambda idx: -crowding_distances[idx])
        selected_indices.extend(remaining[:population_size - len(selected_indices)])

    # 提取新种群、第一前沿的大小和解集
    new_population = [combined[idx][0] for idx in selected_indices]
    first_front_size = len(fronts[0])
    first_front_individuals = [combined[idx][0] for idx in fronts[0]]
    first_front_scores = [combined[idx][1] for idx in fronts[0]]
    new_scores = [combined[idx][1] for idx in selected_indices]

    return new_population, first_front_size, first_front_individuals, first_front_scores, new_scores






def dispatch_all_ccMODE(center, real_path_results, charge_v, num_population, num_cs, cs, num_path, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None):
    def evaluate_complete_solution(sol):
        """
        评估完整解的目标函数值
        """
        f1_value = f1(sol, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice,
                      path_for_choice, anxiety_path_for_choice, num_path, num_cs)
        f2_value = f2(sol, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice,
                      path_for_choice, anxiety_path_for_choice, num_path, num_cs)
        return [f1_value, f2_value]
    print("Starting dispatch_all_ccMODE...")
    od_length, od_wait = process_od_length(real_path_results)

    edge_weights = defaultdict(float)
    node_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[(edge.origin, edge.destination)] = edge.calculate_time()
        edge_weights[(edge.origin, edge.origin)] = 0
        edge_weights[(edge.destination, edge.destination)] = 0
    for node_id, node in center.nodes.items():
        for enter_rid in node.enter:
            for off_rid in node.off:
                node_weights[(center.edges[enter_rid].origin, node_id, center.edges[off_rid].destination)] = node.calculate_wait(enter_rid, off_rid)

    # cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    # anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    print("Processing path and cs results for normal and anxiety...")
    path_for_choice, cs_for_choice = process_path_results(edge_weights, node_weights, OD_ratio, cs, num_path, num_cs)
    anxiety_path_for_choice, anxiety_cs_for_choice = process_path_results(edge_weights, node_weights, anxiety_OD_ratio, cs, num_path, num_cs,0)

    num_od = len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()) if anxiety_OD_ratio else len(OD_ratio)
    F1_cnt = [0 for i in range(num_od)]
    score_cnt = [[]] * num_od
    ref_point = [[0, 0] for _ in range(num_od)]
    col_index = [-1 for i in range(num_od)]
    REP_max = num_population

    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[(edge.origin, edge.destination)] = edge.calculate_time()

    ##初始化所有子种群
    print("Initializing populations...")
    population = []
    for i in range(num_od):
        sub_population = generate_random_individual(num_cs, num_path)
        population.append([i, sub_population])


    ##进化阶段
    for iter in range(max_iter):
        print(f"Iteration {iter + 1}/{max_iter}")

        for i in range(num_od):
            sub_population = population[i][1]
            new_population = []
            for j in range(num_population):
                ind = sub_population[j]
                new_population.append(ind)
                mu = rand_1(j, sub_population)
                off = simple_crossover(ind, mu, num_path)
                new_population.append(off)
            population[i][1] = new_population

        for i in range(num_od):
            sub_population = population[i][1]
            ref_point[i] = [-1, -1]
            if iter == 0:
                for j in range(0, num_population * 2):
                    random_col_index = [random.randint(0, num_population - 1) for _ in range(num_population)]
                    sol = []
                    for k in range(num_od):
                        if k != i:
                            other_sub_population = population[k][1]
                            other_sol = other_sub_population[random_col_index[k]]
                            sol += other_sol
                        else:
                            sol += sub_population[j]
                    point = evaluate_complete_solution(sol)
                    if ref_point[i][0] == -1 or point[0] < ref_point[i][0]:
                        ref_point[i][0] = point[0]
                    if ref_point[i][1] == -1 or point[1] < ref_point[i][1]:
                        ref_point[i][1] = point[1]
                    score_cnt[i].append(point)


            else:
                for j in range(num_population, num_population * 2):
                    sol = []
                    for k in range(num_od):
                        if k != i:
                            other_sub_population = population[k][1]
                            other_sol = other_sub_population[col_index[k]]
                            sol += other_sol
                        else:
                            sol += sub_population[j]
                    point = evaluate_complete_solution(sol)
                    if ref_point[i][0] == -1 or point[0] < ref_point[i][0]:
                        ref_point[i][0] = point[0]
                    if ref_point[i][1] == -1 or point[1] < ref_point[i][1]:
                        ref_point[i][1] = point[1]
                    score_cnt[i].append(point)

            # 在 dispatch_all_ccMODE 中调用
            for i in range(num_od):
                sub_population = population[i][1]
                sub_scores = score_cnt[i]
                new_sub_population, first_front_size, first_front_sols, first_front_scores, new_scores = fast_non_dominated_sort_and_crowding(
                    sub_population, sub_scores, num_population
                )
                population[i][1] = new_sub_population
                score_cnt[i] = new_scores  # 更新得分列表
                F1_cnt[i] = first_front_size

                print(f"子种群 {i} 第一前沿大小: {first_front_size}")
                # 如果需要查看第一前沿解集和指标对
                # print(f"第一前沿解集: {first_front_sols}")
                print(f"第一前沿指标对: {first_front_scores}")

                col_index[i] = find_best_contributor(score_cnt[i], F1_cnt[i])

        if iter == max_iter - 1:
            print("Reached maximum iterations, stopping evolution.")
            final_sol = []
            for i in range(num_od):
                best_idx = col_index[i]
                if best_idx == -1:
                    print(f"Warning: No valid index found for subpopulation {i}. Using first individual as fallback.")
                    best_idx = 0
                final_sol += population[i][1][best_idx]

    print("Evolution completed. Applying best solutions to vehicles.")
    print(f"final_sol: {final_sol}")


    ##最优解应用部分
    v_no = charge_v
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id

    def transform_node_to_edge(path):
        edge_id_path = []
        for i in range(len(path) - 1):
            edge_id_path.append(edge_od_id[(path[i], path[i + 1])])
        return edge_id_path

    for i in range(num_od):
        best_idx = col_index[i]
        if best_idx == -1:
            print(f"Warning: No valid index found for subpopulation {i}. Using first individual as fallback.")
            best_idx = 0
        sol = population[i][1][best_idx]
        od = index_to_od(i, OD_ratio, anxiety_OD_ratio)
        o, d = od
        total_vn = OD_ratio[od] if i < len(OD_ratio.keys()) else anxiety_OD_ratio[od]
        num_this_path = 0

        for j in range(num_cs):
            cs_id = cs_for_choice[od][j]
            for k in range(num_path):
                weight = sol[j * num_path + k]
                num_this_path = 0
                if i < len(OD_ratio.keys()):
                    path = path_for_choice[(o, cs_id, d)][k]['path']
                    num_this_path = int(weight * OD_ratio[od])
                else:
                    path = anxiety_path_for_choice[(o, d, cs_id)][k]['path']
                    num_this_path = int(weight * anxiety_OD_ratio[od])

                if j == num_cs - 1 and k == num_path - 1:
                    num_this_path = total_vn

                for _ in range(num_this_path):
                    vid = v_no[0]
                    vehicle = center.vehicles[vid]
                    if vehicle.origin == o and vehicle.destination == d:
                        print("校对正确")
                    else:
                        print(f"校对错误: 车辆 {vid} 的起点 {vehicle.origin} 和终点 {vehicle.destination} 与OD对 {od} 不匹配")

                    vehicle.charge = (cs_id, 300)
                    vehicle.path = transform_node_to_edge(path)
                    vehicle.road = vehicle.path[0]
                    vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1
                    vehicle.distance = center.edges[vehicle.road].length
                    vehicle.speed = center.edges[vehicle.road].calculate_drive()

                    center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["all"], 1)
                    if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                        center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                            center.edges[vehicle.road].capacity["charge"], 1)
                    center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                        center.edges[vehicle.road].capacity[vehicle.next_road], 1)

                    # print(f"Vehicle {vehicle_id} assigned to CS {cs_id} with path {vehicle.path}")
                    vehicle.drive()

    if len(v_no) > 0:
        print(f"Vehicles remaining: {v_no}")
    else:
        print("All vehicles have been assigned paths and charged.")

