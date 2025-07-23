import operator
import numpy as np
from numba.core.typing.arraydecl import sum_expand

import simuplus
import math
import random
import heapq
import multiprocessing as mp
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import pandapower as pp
import copy
import time
from numba.cpython.randomimpl import permutation_impl
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from collections import defaultdict
import os


def normalize_vector(vector):
    """确保向量元素和为1的精确归一化函数"""
    total = sum(vector)
    if total == 0:
        # 处理全零向量
        return [1.0/len(vector) for _ in range(len(vector))]
    else:
        # 先归一化
        normalized = [v / total for v in vector]
        # 检查是否精确和为1
        actual_sum = sum(normalized)
        if actual_sum != 1.0:
            # 对第一个非零元素进行微调
            for i in range(len(normalized)):
                if normalized[i] > 0:
                    normalized[i] += (1.0 - actual_sum)
                    break
        return normalized


# def process_od_length(real_path_results):
#     od_length = {}
#     for od, result in real_path_results.items():
#         if od[0] == od[1]: continue
#         sum = div = 0
#         for i in range(len(result) - 1):
#             sum += 1 / math.exp(result[i][1] + result[i][2])
#             div += 1
#         if div > 0:
#             sum /= div
#         else:
#             sum = 1 / math.exp(result[0][1] + result[0][2])
#         od_length[od] = sum
#         od_length[(od[0], od[0])] = 0
#         od_length[(od[1], od[1])] = 0
#
#     od_wait = {}
#     for od, result in real_path_results.items():
#         sum = div = 0
#         for i in range(len(result) - 1):
#             sum += 1 / math.exp(result[0][2])
#             div += 1
#         if div > 0:
#             sum /= div
#         else:
#             sum = 1 / math.exp(result[0][2])
#         od_wait[od] = sum
#     return od_length, od_wait


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

    参数:
        num_cs: 充电站数量
        num_path: 每个充电站的路径数量
        seed: 随机数种子，默认为None

    返回:
        list: 包含num_cs个子向量的列表，每个子向量长度为num_path
    """

    # random.seed(random.randint(1, 114514))

    individual = []
    weight = generate_random_1(num_cs)
    for i in range(num_cs):
        sub_vector = generate_random_1(num_path)
        subs = [x * weight[i] for x in sub_vector]
        ss = sum(subs)
        for sub in subs:
            individual.append(sub / ss)

    # 重置随机种子，避免影响其他随机操作
    # if seed is not None:
    #     random.seed()

    return normalize_vector(individual)

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
                    # print(kk, path_for_choice[(o, cs_id, d)][kk]['path'])
                    # print(f"+{weight * OD_ratio[od] * path_for_choice[(o, cs_id, d)][kk]['total_time']}, weight:{weight}, OD_ratio:{OD_ratio[od]}, path_time:{path_for_choice[(o, cs_id, d)][kk]['total_time']}")

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
    """
    # 处理特殊情况
    if start == stop and stop == end:
        return [{
            'path': [start],
            'travel_time': 0,
            'wait_time': 0,
            'total_time': 0,
            'first_part_travel_time': 0,
            'second_part_travel_time': 0,
            'first_part_wait_time': 0,
            'second_part_wait_time': 0,
        }]


    # print(f"等待信息：{node_weights}")

    # 构建邻接表
    adj_list = defaultdict(list)
    for (src, dst), weight in graph.items():
        adj_list[src].append((dst, weight))

    # 计算路径的等待时间和行驶时间
    def calculate_path_metrics(path):
        travel_time = 0
        wait_time = 0

        # 计算行驶时间
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            if edge_key in graph:
                travel_time += graph[edge_key]
            else:
                print(f"警告: 边 {edge_key} 不在图中")
                return None

        # 计算等待时间 - 考虑三元组 (prev, current, next)
        for i in range(1, len(path) - 1):
            wait_key = (path[i - 1], path[i], path[i + 1])
            if wait_key in node_weights:
                wait_time += node_weights[wait_key]
                # print(f"节点 {path[i]} 的等待时间: {node_weights[wait_key]}")
            else:
                print(f"警告: 等待时间键 {wait_key} 不在node_weights中")

        return {
            'path': path,
            'travel_time': travel_time,
            'wait_time': wait_time,
            'total_time': travel_time + wait_time
        }

    # 单源最短路径算法
    def dijkstra(source, target, excluded_edges=None):
        if excluded_edges is None:
            excluded_edges = set()

        dist = {node: float('inf') for node in adj_list}
        dist[source] = 0
        prev = {node: None for node in adj_list}
        visited = set()
        pq = [(0, source)]

        while pq:
            d, node = heapq.heappop(pq)

            if node in visited:
                continue

            visited.add(node)

            if node == target:
                break

            for neighbor, weight in adj_list[node]:
                if (node, neighbor) in excluded_edges:
                    continue

                new_dist = dist[node] + weight

                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

        # 重建路径
        if dist[target] == float('inf'):
            return None

        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()

        return path

    # Yen算法实现
    def find_k_paths(source, target, max_k):
        # 找到第一条最短路径
        first_path = dijkstra(source, target)

        if not first_path:
            return []

        path_metrics = calculate_path_metrics(first_path)
        if not path_metrics:
            return []

        A = [path_metrics]  # 结果集
        B = []  # 候选路径集

        # 记录已经找到的路径，避免重复
        found_paths = {str(first_path)}

        # Yen算法主循环
        for k_idx in range(1, max_k):
            prev_path = A[-1]['path']

            # 对前一条路径的每个节点进行偏离尝试
            for i in range(len(prev_path) - 1):
                # 偏离点
                spur_node = prev_path[i]
                # 根路径
                root_path = prev_path[:i + 1]

                # 需要临时排除的边
                excluded_edges = set()

                # 排除已有路径中会导致重复的边
                for path_dict in A:
                    p = path_dict['path']
                    if len(p) > i and p[:i + 1] == root_path:
                        if i < len(p) - 1:
                            excluded_edges.add((p[i], p[i + 1]))

                # 排除根路径中的节点(除了spur_node)以避免环路
                for j in range(i):
                    root_node = root_path[j]
                    for neighbor, _ in adj_list[root_node]:
                        excluded_edges.add((root_node, neighbor))

                # 从偏离点找新路径
                spur_path = dijkstra(spur_node, target, excluded_edges)

                if not spur_path:
                    continue

                # 完整路径 = 根路径 + 偏离路径(不重复spur_node)
                if len(root_path) > 0 and len(spur_path) > 0 and root_path[-1] == spur_path[0]:
                    total_path = root_path + spur_path[1:]
                else:
                    total_path = root_path + spur_path[1:]

                # 计算新路径的度量
                path_metrics = calculate_path_metrics(total_path)
                if not path_metrics:
                    continue

                # 检查是否是新路径
                path_str = str(total_path)
                if path_str not in found_paths:
                    found_paths.add(path_str)
                    B.append(path_metrics)

            if not B:
                # 没有更多路径可找
                break

            # 按总时间排序
            B.sort(key=lambda x: x['total_time'])

            # 将最短的候选路径添加到结果集
            A.append(B[0])
            B.pop(0)

        return A

    # 处理各种情况
    combined_paths = []

    if start == stop:
        # 起点等于中间点
        paths_from_stop = find_k_paths(stop, end, k)
        for path in paths_from_stop:
            combined_path = {
                'path': path['path'],
                'travel_time': path['travel_time'],
                'wait_time': path['wait_time'],
                'total_time': path['total_time'],
                'first_part_travel_time': 0,
                'second_part_travel_time': path['travel_time'],
                'first_part_wait_time': 0,
                'second_part_wait_time': path['wait_time'],
            }
            combined_paths.append(combined_path)

    elif stop == end:
        # 中间点等于终点
        paths_to_stop = find_k_paths(start, stop, k)
        for path in paths_to_stop:
            combined_path = {
                'path': path['path'],
                'travel_time': path['travel_time'],
                'wait_time': path['wait_time'],
                'total_time': path['total_time'],
                'first_part_travel_time': path['travel_time'],
                'second_part_travel_time': 0,
                'first_part_wait_time': path['wait_time'],
                'second_part_wait_time': 0,
            }
            combined_paths.append(combined_path)

    else:
        # 常规情况：寻找start→stop和stop→end的路径
        paths_to_stop = find_k_paths(start, stop, k)
        paths_from_stop = find_k_paths(stop, end, k)

        # 打印调试信息
        # print(f"从 {start} 到 {stop} 找到 {len(paths_to_stop)} 条路径")
        # for i, p in enumerate(paths_to_stop):
        #     print(f"  路径 {i + 1}: {p['path']}, 总时间: {p['total_time']}")
        #
        # print(f"从 {stop} 到 {end} 找到 {len(paths_from_stop)} 条路径")
        # for i, p in enumerate(paths_from_stop):
        #     print(f"  路径 {i + 1}: {p['path']}, 总时间: {p['total_time']}")

        # 组合路径
        for path1 in paths_to_stop:
            for path2 in paths_from_stop:
                # 合并路径（去掉重复的中间节点）
                combined_path = path1['path'][:-1] + path2['path']

                # 计算第一部分和第二部分的行驶时间和等待时间
                first_part_travel_time = path1['travel_time']
                second_part_travel_time = path2['travel_time']
                first_part_wait_time = path1['wait_time']
                second_part_wait_time = path2['wait_time']

                # 计算总行驶时间和总等待时间
                travel_time = first_part_travel_time + second_part_travel_time
                wait_time = first_part_wait_time + second_part_wait_time

                combined_path_dict = {
                    'path': combined_path,
                    'travel_time': travel_time,
                    'wait_time': wait_time,
                    'total_time': travel_time + wait_time,
                    'first_part_travel_time': first_part_travel_time,
                    'second_part_travel_time': second_part_travel_time,
                    'first_part_wait_time': first_part_wait_time,
                    'second_part_wait_time': second_part_wait_time,
                }
                combined_paths.append(combined_path_dict)

    # 确保路径多样性
    unique_paths = []
    path_strings = set()

    for path in combined_paths:
        path_str = str(path['path'])
        if path_str not in path_strings:
            path_strings.add(path_str)
            unique_paths.append(path)

    # 按总时间排序并返回
    unique_paths.sort(key=lambda x: x['total_time'])

    # print(f"找到{len(unique_paths)}条唯一路径，返回前{min(k, len(unique_paths))}条")
    # for i, p in enumerate(unique_paths[:k]):
    #     print(f"  最终路径 {i + 1}: {p['path']}, 总时间: {p['total_time']}")

    return unique_paths[:k]


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
                    # print(f"存在路径{paths},共{len(paths)}条")
                    utility = 0
                    for cnt in range(k):
                        utility += math.exp(-0.1*(paths[cnt]['total_time'])) / k
                else:
                    print(f"Warning: No paths found for OD {od} via CS {cs_id}. Using negative infinity as utility.{anxiety}")
                    utility = -1
                path_for_choice[(o, cs_id, d)] = paths
            else:
                paths = find_k_shortest_paths_via_stop(edge_weights, node_weights, o, d, cs_id, k)
                if paths:
                    utility = 0
                    for cnt in range(k):
                        utility += math.exp(-0.1 * (paths[cnt]['total_time'])) / k
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
    mu = [ind[i] + beta * (population[indices[0]][i] - population[indices[1]][i])
            for i in range(len(ind))]
    for r in range(len(mu)):
        if mu[r] < 0:
            mu[r] *= -1.0

    return normalize_vector(mu)


def rand_2(index, population, beta1=0.4, beta2=0.4):
    ind = population[index]
    indices = select_random_indices(len(ind), 4, index)

    mu = [ind[i] + beta1 * (population[indices[0]][i] - population[indices[1]][i])
            + beta2 * (population[indices[2]][i] - population[indices[3]][i])
            for i in range(len(ind))]

    for r in range(len(mu)):
        if mu[r] < 0:
            mu[r] *= -1.0

    return normalize_vector(mu)

def current_to_best_1(index, population, best_individual, beta=0.4):
    """
    当前个体与最优个体之间的差异
    """
    ind = population[index]
    indices = select_random_indices(len(ind), 2, index)
    mu = [ind[i] + beta * (best_individual[i] - ind[i])
          + beta * (population[indices[0]][i] - population[indices[1]][i])  for i in range(len(ind))]

    for r in range(len(mu)):
        if mu[r] < 0:
            mu[r] *= -1.0

    return normalize_vector(mu)


def simple_crossover(ind, mu, num_gene, mu_rate=0.5):
    """
    简单交叉操作，随机选择一个位置进行交叉
    """
    i = int(len(ind) / num_gene)
    off = []
    for j in range(i):
        r = random.random()
        if r < mu_rate:
            off+=mu[j * num_gene:(j + 1) * num_gene]
        else:
            off+=ind[j * num_gene:(j + 1) * num_gene]
    if off == ind:
        return normalize_vector(mu)
    else:
        return normalize_vector(off)




# def fast_non_dominated_sort_and_crowding(population, scores, population_size):
#     """
#     对种群进行快速非支配排序和拥挤度排序，选择前population_size个解
#
#     参数:
#         population: 种群，每个元素是一个个体
#         scores: 每个个体对应的目标函数值列表，格式为 [[f1_1, f2_1], [f1_2, f2_2], ...]
#         population_size: 返回的种群大小
#
#     返回:
#         tuple: (新种群, 前沿数量, 前沿解集, 前沿解集的指标对)
#     """
#     # 合并种群和得分为(个体, 得分)对
#     # print(f"scores: {scores}")
#     combined = list(zip(population, scores))
#
#     # 计算支配关系
#     domination_counts = [0 for _ in range(len(combined))]   # 支配当前个体的其他个体数量
#     dominated_sets = [[] for _ in range(len(combined))]  # 被当前个体支配的个体列表
#
#     # 快速非支配排序
#     for i in range(len(combined)):
#         for j in range(len(combined)):
#             if i == j:
#                 continue
#
#             # 检查i是否支配j
#             if (combined[i][1][0] <= combined[j][1][0] and combined[i][1][1] <= combined[j][1][1]) and \
#                     (combined[i][1][0] < combined[j][1][0] or combined[i][1][1] < combined[j][1][1]):
#                 dominated_sets[i].append(j)
#             # 检查j是否支配i
#             elif (combined[j][1][0] <= combined[i][1][0] and combined[j][1][1] <= combined[i][1][1]) and \
#                     (combined[j][1][0] < combined[i][1][0] or combined[j][1][1] < combined[i][1][1]):
#                 domination_counts[i] += 1
#
#     # 按前沿对个体进行分组
#     fronts = []
#     current_front = []
#
#     # 找到第一个前沿
#     for i in range(len(combined)):
#         if domination_counts[i] == 0:
#             current_front.append(i)
#
#     fronts.append(current_front)
#
#     # 找到其余前沿
#     i = 0
#     while i < len(fronts):
#         next_front = []
#         for j in fronts[i]:
#             for k in dominated_sets[j]:
#                 domination_counts[k] -= 1
#                 if domination_counts[k] == 0:
#                     next_front.append(k)
#         i += 1
#         if next_front:
#             fronts.append(next_front)
#
#     # 计算拥挤度
#     def calculate_crowding_distance(front_indices):
#         if len(front_indices) <= 2:
#             return {idx: float('inf') for idx in front_indices}
#
#         distances = {idx: 0 for idx in front_indices}
#
#         # 对每个目标函数排序并计算拥挤度
#         for obj in range(2):  # 假设有2个目标函数
#             sorted_front = sorted(front_indices, key=lambda idx: combined[idx][1][obj])
#
#             # 设置边界点的拥挤度为无穷大
#             distances[sorted_front[0]] = float('inf')
#             distances[sorted_front[-1]] = float('inf')
#
#             # 计算中间点的拥挤度
#             obj_range = combined[sorted_front[-1]][1][obj] - combined[sorted_front[0]][1][obj]
#             if obj_range == 0:
#                 continue  # 避免除以零
#
#             for i in range(1, len(sorted_front) - 1):
#                 distances[sorted_front[i]] += (combined[sorted_front[i + 1]][1][obj] -
#                                                combined[sorted_front[i - 1]][1][obj]) / obj_range
#
#         return distances
#
#     # 为每个前沿计算拥挤度
#     crowding_distances = {}
#     for front in fronts:
#         distances = calculate_crowding_distance(front)
#         crowding_distances.update(distances)
#
#     # 按前沿和拥挤度选择新种群
#     selected_indices = []
#     i = 0
#     while len(selected_indices) + len(fronts[i]) <= population_size:
#         selected_indices.extend(fronts[i])
#         i += 1
#         if i >= len(fronts):
#             break
#
#     # 如果还需要更多个体，则根据拥挤度选择
#     if len(selected_indices) < population_size and i < len(fronts):
#         remaining = sorted(fronts[i], key=lambda idx: -crowding_distances[idx])
#         selected_indices.extend(remaining[:population_size - len(selected_indices)])
#
#     # 提取新种群、第一前沿的大小和解集
#     new_population = [combined[idx][0] for idx in selected_indices]
#     first_front_size = len(fronts[0])
#     first_front_individuals = [combined[idx][0] for idx in fronts[0]]
#     first_front_scores = [combined[idx][1] for idx in fronts[0]]
#     new_scores = [combined[idx][1] for idx in selected_indices]
#
#     return new_population, first_front_size, first_front_individuals, first_front_scores, new_scores

def fast_non_dominated_sort_and_crowding(population, scores, population_size):
    """
    对种群进行快速非支配排序和拥挤度排序，选择前population_size个解
    当第一前沿个体数量超过种群大小时，仅输出保留的population_size个个体与对应分数作为第一前沿

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
    domination_counts = [0 for _ in range(len(combined))]  # 支配当前个体的其他个体数量
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

    # 如果第一前沿大小已经超过了population_size，就直接对第一前沿进行拥挤度排序
    if len(current_front) >= population_size:
        # 计算拥挤度
        crowding_distances = {}
        front_points = [combined[i][1] for i in current_front]

        # 对每个目标函数进行排序
        sorted_indices = []
        num_obj = len(scores[0])
        for obj in range(num_obj):  # 假设有2个目标函数
            # 按照第obj个目标函数值排序
            sorted_by_obj = sorted(range(len(current_front)), key=lambda i: front_points[i][obj])
            sorted_indices.append(sorted_by_obj)

        # 初始化拥挤度
        for i in current_front:
            crowding_distances[i] = 0

        # 计算拥挤度
        for obj in range(2):
            # 获取最小和最大值，避免除以零
            obj_min = front_points[sorted_indices[obj][0]][obj]
            obj_max = front_points[sorted_indices[obj][-1]][obj]
            range_obj = obj_max - obj_min if obj_max > obj_min else 1

            # 边界点拥挤度为无穷大
            crowding_distances[current_front[sorted_indices[obj][0]]] = float('inf')
            crowding_distances[current_front[sorted_indices[obj][-1]]] = float('inf')

            # 计算中间点的拥挤度
            for i in range(1, len(sorted_indices[obj]) - 1):
                idx = sorted_indices[obj][i]
                next_idx = sorted_indices[obj][i + 1]
                prev_idx = sorted_indices[obj][i - 1]

                # 计算当前点的拥挤度
                if range_obj > 0:
                    crowding_distances[current_front[idx]] += (
                                                                      front_points[next_idx][obj] -
                                                                      front_points[prev_idx][obj]
                                                              ) / range_obj

        # 根据拥挤度排序并选择前population_size个个体
        current_front.sort(key=lambda i: crowding_distances[i], reverse=True)
        selected_indices = current_front[:population_size]

        # 提取最终种群
        new_population = [combined[idx][0] for idx in selected_indices]
        first_front_size = population_size
        first_front_individuals = new_population
        first_front_scores = [combined[idx][1] for idx in selected_indices]
        new_scores = first_front_scores

        return new_population, first_front_size, first_front_individuals, first_front_scores, new_scores

    # 如果第一前沿不够，则继续计算后续前沿
    fronts.append(current_front)

    # 找到其余前沿
    i = 0
    while i < len(fronts):
        next_front = []
        for idx in fronts[i]:
            for j in dominated_sets[idx]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        i += 1

    # 计算拥挤度
    def calculate_crowding_distance(front_indices):
        if len(front_indices) <= 2:
            return {idx: float('inf') for idx in front_indices}

        distances = {idx: 0 for idx in front_indices}
        front_scores = [combined[idx][1] for idx in front_indices]

        # 对每个目标进行处理
        for obj_idx in range(2):  # 假设有2个目标
            # 按当前目标排序
            sorted_indices = sorted(range(len(front_indices)),
                                    key=lambda i: front_scores[i][obj_idx])
            sorted_front = [front_indices[i] for i in sorted_indices]

            # 边界点的拥挤度设为无穷大
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # 归一化因子
            obj_range = front_scores[sorted_indices[-1]][obj_idx] - front_scores[sorted_indices[0]][obj_idx]
            if obj_range == 0:
                continue  # 如果所有值相等，则跳过

            # 计算中间点的拥挤度
            for i in range(1, len(sorted_front) - 1):
                prev_score = front_scores[sorted_indices[i - 1]][obj_idx]
                next_score = front_scores[sorted_indices[i + 1]][obj_idx]
                distances[sorted_front[i]] += (next_score - prev_score) / obj_range

        return distances

    # 为每个前沿计算拥挤度
    crowding_distances = {}
    for front in fronts:
        front_distances = calculate_crowding_distance(front)
        crowding_distances.update(front_distances)

    # 按前沿和拥挤度选择新种群
    selected_indices = []
    i = 0
    while len(selected_indices) + len(fronts[i]) <= population_size:
        selected_indices.extend(fronts[i])
        i += 1

    # 如果还需要更多个体，则根据拥挤度选择
    if len(selected_indices) < population_size and i < len(fronts):
        remaining = population_size - len(selected_indices)
        # 按拥挤度排序
        sorted_front = sorted(fronts[i], key=lambda idx: crowding_distances[idx], reverse=True)
        selected_indices.extend(sorted_front[:remaining])

    # 提取新种群、第一前沿的大小和解集
    new_population = [combined[idx][0] for idx in selected_indices]

    # 确定第一前沿的个体
    if len(fronts[0]) <= population_size:
        # 如果第一前沿小于等于种群大小，则使用完整的第一前沿
        first_front_size = len(fronts[0])
        first_front_individuals = [combined[idx][0] for idx in fronts[0]]
        first_front_scores = [combined[idx][1] for idx in fronts[0]]
    else:
        # 如果第一前沿大于种群大小，则只使用保留的个体
        first_front_indices = [idx for idx in selected_indices if idx in fronts[0]]
        first_front_size = len(first_front_indices)
        first_front_individuals = [combined[idx][0] for idx in first_front_indices]
        first_front_scores = [combined[idx][1] for idx in first_front_indices]

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

        # print(f"Evaluating solution: {sol}")
        # print(f"Evaluating solution: f1={f1_value}, f2={f2_value}")
        return [f1_value, f2_value]
    print("Starting dispatch_all_ccMODE...")
    # od_length, od_wait = process_od_length(real_path_results)

    # print(f"lmp_dict: {lmp_dict}")
    sum_lmp = sum(lmp_dict.values())
    for key in lmp_dict:
        lmp_dict[key] /= sum_lmp
    # print(f"lmp_dict: {lmp_dict}")

    print("Calculating edge and node weights...")
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
    # print(f"path_for_choice: {path_for_choice}")

    num_od = len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()) if anxiety_OD_ratio else len(OD_ratio)
    F1_cnt = [0 for _ in range(num_od)]
    score_cnt = [[] for _ in range(num_od)]
    ref_point = [[0, 0] for _ in range(num_od)]
    col_index = [-1 for _ in range(num_od)]
    REP_max = num_population

    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[(edge.origin, edge.destination)] = edge.calculate_time()

    ##初始化所有子种群
    print("Initializing populations...")
    population = []
    for i in range(num_od):
        sub_population = []
        for j in range(num_population):
            sub_population.append(generate_random_individual(num_cs, num_path))
        population.append([i, sub_population])
        # print(f"子种群 {i} 初始化完成，{sub_population}")


    ##进化阶段
    for iter in range(max_iter):
        print(f"Iteration {iter + 1}/{max_iter}")

        for i in range(num_od):
            new_population = population[i][1].copy()
            for j in range(num_population):
                ind = new_population[j]
                mu = rand_1(j, new_population)
                # print(f"ind={ind}")
                # print(f"mu={mu}")
                off = simple_crossover(ind, mu, num_path)
                new_population.append(off)
            population[i][1] = new_population

        for i in range(num_od):
            sub_population = population[i][1]
            print(f"子种群 {i} 的大小: {len(sub_population)}")
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
    v_no = charge_v[0::2] + charge_v[1::2]  # 假设车辆编号是偶数和奇数交替的
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id

    def transform_node_to_edge(path):
        edge_id_path = []
        for i in range(len(path) - 1):
            edge_id_path.append(edge_od_id[(path[i], path[i + 1])])
        return edge_id_path

    for i in range(num_od):
        print(f"Processing subpopulation {i} ")
        best_idx = col_index[i]
        if best_idx == -1:
            print(f"Warning: No valid index found for subpopulation {i}. Using first individual as fallback.")
            best_idx = 0
        sol = population[i][1][best_idx]
        # print(sum(sol))
        od = index_to_od(i, OD_ratio, anxiety_OD_ratio)
        o, d = od
        total_vn = OD_ratio[od] if i < len(OD_ratio.keys()) else anxiety_OD_ratio[od]

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
                    # if vehicle.origin == o and vehicle.destination == d:
                    #     print(f"校对正确：车辆 {vid} 的起点 {vehicle.origin} 和终点 {vehicle.destination} 与OD对 {od} 匹配")
                    # else:
                    #     print(f"校对错误：车辆 {vid} 的起点 {vehicle.origin} 和终点 {vehicle.destination} 与OD对 {od} 不匹配")

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

                    v_no.pop(0)
                    total_vn -= 1
                    # print(f"total_vn: {total_vn}, i:{i}, j: {j}, k: {k}")
                    # print(f"Vehicle {vehicle_id} assigned to CS {cs_id} with path {vehicle.path}")
                    vehicle.drive()
                    if total_vn == 0: break

                if total_vn == 0: break

    if len(v_no) > 0:
        print(f"Vehicles remaining: {v_no}")
    else:
        print("All vehicles have been assigned paths and charged.")























def initialize_sub_population(i, num_population, num_cs, num_path):
    """为指定索引 i 初始化子种群"""
    sub_population = []
    for j in range(num_population):
        sub_population.append(generate_random_individual(num_cs, num_path))
    return [i, sub_population]


def evolve_sub_population(i, sub_population, num_path):
    """并行处理单个子种群的进化操作"""
    new_population = sub_population.copy()
    num_population = len(sub_population)

    for j in range(num_population):
        ind = new_population[j]
        mu = rand_2(j, new_population)
        off = simple_crossover(ind, mu, num_path)
        new_population.append(off)

    return [i, new_population]


def evaluate_single_solution(i, j, sub_population, population, col_index, iter, num_population, num_od,
                             lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus,
                             cs_for_choice, anxiety_cs_for_choice,
                             path_for_choice, anxiety_path_for_choice, num_path, num_cs):
    """评估单个解"""
    sol = []
    # 构建完整解
    if iter == 0:
        # 随机选择
        random_col_index = [random.randint(0, num_population - 1) for _ in range(num_od)]
        for k in range(num_od):
            if k != i:
                sol += population[k][1][random_col_index[k]]
            else:
                sol += sub_population[j % len(sub_population)]
    else:
        # 使用最佳解
        for k in range(num_od):
            if k != i:
                # sol += population[k][1][col_index[k]]
                sol += col_index[k]
            else:
                sol += sub_population[j % len(sub_population)]

    # 直接计算目标函数
    f1_value = f1(sol, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice,
                  path_for_choice, anxiety_path_for_choice, num_path, num_cs)
    f2_value = f2(sol, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice,
                  path_for_choice, anxiety_path_for_choice, num_path, num_cs)
    point = [f1_value, f2_value]

    return j, point


def evaluate_population_solutions(i, sub_population, population, col_index, iter, num_population, num_od,
                                  lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus,
                                  cs_for_choice, anxiety_cs_for_choice,
                                  path_for_choice, anxiety_path_for_choice, num_path, num_cs, multi_evo = False):
    """并行处理单个子种群的解评估"""
    print(f"评估子种群 {i}...")
    local_ref_point = [-1, -1]
    local_scores = []

    index_range = range(0, num_population * 2)

    # 使用内部线程池并行评估每个解
    inner_scores = []

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(index_range))) as inner_executor:
        inner_futures = []

        for j in index_range:
            inner_futures.append(
                inner_executor.submit(
                    evaluate_single_solution, i, j, sub_population, population,
                    col_index, iter, num_population, num_od,
                    lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus,
                    cs_for_choice, anxiety_cs_for_choice,
                    path_for_choice, anxiety_path_for_choice, num_path, num_cs
                )
            )

        # 收集内部评估结果
        completed_inner = 0
        total_inner = len(inner_futures)
        for future in concurrent.futures.as_completed(inner_futures):
            j, point = future.result()
            inner_scores.append((j, point))
            completed_inner += 1
            # if completed_inner % 10 == 0:  # 每10个结果打印一次进度
            #     print(f"子种群 {i} 内部评估进度: {completed_inner}/{total_inner}")

    # 处理收集到的结果
    inner_scores.sort(key=lambda x: x[0])  # 按索引排序
    for _, point in inner_scores:
        local_scores.append(point)

        # 更新参考点
        if local_ref_point[0] == -1 or point[0] < local_ref_point[0]:
            local_ref_point[0] = point[0]
        if local_ref_point[1] == -1 or point[1] < local_ref_point[1]:
            local_ref_point[1] = point[1]

    # print(f"子种群 {i} 评估完成，共评估 {len(local_scores)} 个解")
    return i, local_ref_point, local_scores


# def evaluate_population_solutions(i, sub_population, population, col_list, iter, num_population, num_od,
#                                   lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus,
#                                   cs_for_choice, anxiety_cs_for_choice,
#                                   path_for_choice, anxiety_path_for_choice, num_path, num_cs, multi_evo=False):
#     """并行处理单个子种群的解评估，使用进程池代替线程池"""
#     print(f"评估子种群 {i}...")
#     local_ref_point = [-1, -1]
#     local_scores = []
#
#     # 确定评估范围
#     if iter == 0:
#         # 子种群进化期间只评估新个体
#         index_range = range(num_population, num_population * 2)
#     else:
#         # 协作期间评估所有个体
#         index_range = range(0, num_population * 2)
#
#     # 准备要评估的个体列表
#     eval_tasks = []
#     for j in index_range:
#         if j < len(sub_population):
#             eval_tasks.append((i, j, sub_population, population, col_list, iter,
#                                num_population, num_od, lmp_dict, OD_ratio,
#                                anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
#                                anxiety_cs_for_choice, path_for_choice,
#                                anxiety_path_for_choice, num_path, num_cs))
#
#     # 使用进程池并行评估每个解
#     inner_scores = []
#     total_inner = len(eval_tasks)
#
#     process_count = 2
#
#     with ProcessPoolExecutor(max_workers=process_count) as process_executor:
#         # 提交所有任务
#         futures = [process_executor.submit(evaluate_single_solution, *task) for task in eval_tasks]
#
#         # 收集进程评估结果
#         completed_inner = 0
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 j, point = future.result()
#                 inner_scores.append((j, point))
#                 completed_inner += 1
#                 if completed_inner % 10 == 0 or completed_inner == total_inner:
#                     print(f"  子种群 {i}: 已评估 {completed_inner}/{total_inner} 个解")
#             except Exception as e:
#                 print(f"评估过程中出错: {e}")
#
#     # 处理收集到的结果
#     inner_scores.sort(key=lambda x: x[0])  # 按索引排序
#     for _, point in inner_scores:
#         local_scores.append(point)
#
#         # 更新参考点
#         if local_ref_point[0] == -1 or point[0] > local_ref_point[0]:
#             local_ref_point[0] = point[0]
#         if local_ref_point[1] == -1 or point[1] > local_ref_point[1]:
#             local_ref_point[1] = point[1]
#
#     return i, local_ref_point, local_scores









def process_subpopulation(i, sub_population, sub_scores, num_population):
    """并行处理单个子种群的非支配排序和拥挤度计算"""
    new_sub_population, first_front_size, first_front_sols, first_front_scores, new_scores = fast_non_dominated_sort_and_crowding(
        sub_population, sub_scores, num_population
    )

    print(f"子种群 {i} 第一前沿大小: {first_front_size}")
    print(f"第一前沿指标对: {first_front_scores}")

    best_contributor_idx = find_best_contributor(new_scores, first_front_size)

    return i, new_sub_population, new_scores, first_front_size, best_contributor_idx








def dispatch_all_ccMODE_parallel(center, real_path_results, charge_v, num_population, num_cs, cs, num_path, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None):
    evol_per_col = 5

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
    # od_length, od_wait = process_od_length(real_path_results)

    edge_weights = defaultdict(float)
    node_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[(edge.origin, edge.destination)] = edge.calculate_time()
        edge_weights[(edge.origin, edge.origin)] = 0.0
        edge_weights[(edge.destination, edge.destination)] = 0.0
    for node_id, node in center.nodes.items():
        for enter_rid in node.enter:
            for off_rid in node.off:
                node_weights[(center.edges[enter_rid].origin, node_id, center.edges[off_rid].destination)] = node.calculate_wait(enter_rid, off_rid)

    print(f"calculated edge_weights: {edge_weights}")
    print(f"calculated node_weights: {node_weights}")

    # cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    # anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    print("Processing path and cs results for normal and anxiety...")
    path_for_choice, cs_for_choice = process_path_results(edge_weights, node_weights, OD_ratio, cs, num_path, num_cs)
    anxiety_path_for_choice, anxiety_cs_for_choice = process_path_results(edge_weights, node_weights, anxiety_OD_ratio, cs, num_path, num_cs,0)

    num_od = len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()) if anxiety_OD_ratio else len(OD_ratio)
    F1_cnt = [0 for _ in range(num_od)]
    score_cnt = [[] for _ in range(num_od)]
    ref_point = [[0, 0] for _ in range(num_od)]
    col_index = [None for _ in range(num_od)]  #修改过，存的是具体的协作者个体
    REP_max = num_population

    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[(edge.origin, edge.destination)] = edge.calculate_time()

    cpu_cnt = os.cpu_count()
    print(f"Using {cpu_cnt} CPU cores for parallel processing.")
    # 并行初始化所有子种群
    print("Initializing populations in parallel...")
    population = [None for _ in range(num_od)]  # 预先创建固定大小的列表


    # 使用进程池并行初始化子种群
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), num_od)) as executor:
        init_results = list(executor.map(
            initialize_sub_population,
            range(num_od),
            [num_population] * num_od,
            [num_cs] * num_od,
            [num_path] * num_od
        ))

    # 按索引排序以保持顺序
    init_results.sort(key=lambda x: x[0])
    population = init_results
    # for ind in population:
    #     print(ind[0], end=' ')


    ##进化阶段
    for iter in range(max_iter * evol_per_col):
        print(f"Iteration {iter + 1}/{max_iter}")

        # 在每次迭代中并行处理所有子种群的进化
        print("Evolving sub-populations in parallel...")
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), num_od)) as executor:
            # 提交所有子种群的进化任务
            future_to_index = {
                executor.submit(evolve_sub_population, i, population[i][1], num_path): i
                for i in range(num_od)
            }

            # 收集结果并更新种群
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                result = future.result()
                population[i][1] = result[1]

        # 在主循环中使用
        print("Evaluating population solutions in parallel...")

        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), num_od)) as executor:
            futures = []
            for i in range(num_od):
                sub_population = population[i][1]
                futures.append(executor.submit(
                    evaluate_population_solutions, i, sub_population,
                    population, col_index, iter, num_population, num_od,
                    lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus,
                    cs_for_choice, anxiety_cs_for_choice,
                    path_for_choice, anxiety_path_for_choice, num_path, num_cs))

            # 收集评估结果
            for future in concurrent.futures.as_completed(futures):
                i, local_ref, local_scores = future.result()
                ref_point[i] = local_ref
                score_cnt[i].extend(local_scores)


        # 在主循环中并行处理所有子种群的非支配排序和拥挤度计算
        print("Processing sub-populations for non-dominated sorting and crowding in parallel...")
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), num_od)) as executor:
            process_results = list(executor.map(
                process_subpopulation,
                range(num_od),
                [sub_pop[1] for sub_pop in population],
                score_cnt,
                [num_population] * num_od
            ))

        # 更新种群和得分
        for i, new_sub_pop, new_scores, first_front_size, best_idx in process_results:
            population[i][1] = new_sub_pop
            score_cnt[i] = new_scores
            F1_cnt[i] = first_front_size
            if iter % evol_per_col == 0:
                if i == 0: print("Updating collaboration indices...")
                col_index[i] = new_sub_pop[best_idx]

        if iter == max_iter - 1:
            print("Reached maximum iterations, stopping evolution.")
            final_sol = []
            for i in range(num_od):
                if col_index[i] == None:
                    print(f"Warning: No valid index found for subpopulation {i}. Using first individual as fallback.")
                final_sol += col_index[i]

    print("Evolution completed. Applying best solutions to vehicles.")
    print(f"final_sol: {final_sol}")


    ##最优解应用部分
    drive_cnt = 0
    v_no = charge_v[0::2] + charge_v[1::2]  # 假设车辆编号是偶数和奇数交替的
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id

    def transform_node_to_edge(path):
        edge_id_path = []
        for i in range(len(path) - 1):
            edge_id_path.append(edge_od_id[(path[i], path[i + 1])])
        return edge_id_path

    for i in range(num_od):
        print(f"Processing subpopulation {i} with best index")
        if col_index[i] == None:
            print(f"Warning: No valid index found for subpopulation {i}. Using first individual as fallback.")
        sol = col_index[i]
        print(sum(sol))
        od = index_to_od(i, OD_ratio, anxiety_OD_ratio)
        o, d = od
        total_vn = OD_ratio[od] if i < len(OD_ratio.keys()) else anxiety_OD_ratio[od]

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
                        print(f"校对正确：车辆 {vid} 的起点 {vehicle.origin} 和终点 {vehicle.destination} 与OD对 {od} 匹配")
                    else:
                        print(f"校对错误：车辆 {vid} 的起点 {vehicle.origin} 和终点 {vehicle.destination} 与OD对 {od} 不匹配")

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

                    v_no.pop(0)
                    total_vn -= 1
                    print(f"total_vn: {total_vn}, i:{i}, j: {j}, k: {k}")
                    drive_cnt += 1
                    # print(f"Vehicle {vehicle_id} assigned to CS {cs_id} with path {vehicle.path}")
                    vehicle.drive()
                    if total_vn == 0: break

                if total_vn == 0: break

    if len(v_no) > 0:
        print(f"Vehicles remaining: {v_no}")
    else:
        print(f"{drive_cnt} vehicles have been assigned paths and charged.")

