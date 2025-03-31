import numpy as np
import simuplus
import math
import random
import heapq
import multiprocessing as mp
import networkx as nx
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import pandapower as pp


#NSGA2编码：对每辆车预处理车辆没电前可到达并且O-站和站-D的距离之和最短的2^k个站点，排序后按照二进制编码其下标，每辆车的编码长度为k，总长就是车数*k
#

def process_od_length(real_path_results):
    od_length = {}
    for od, result in real_path_results.items():
        if od[0] == od[1]: continue
        sum = div = 0
        for i in range(len(result) - 1):
            sum += (result[i][1] + result[i][2]) / (i + 1)
            div += 1 / (i + 1)
        if div > 0:
            sum /= div
        else:
            sum = result[0][1] + result[0][2]
        od_length[od] = sum
        od_length[(od[0], od[0])] = 0
        od_length[(od[1], od[1])] = 0

    od_wait = {}
    for od, result in real_path_results.items():
        sum = div = 0
        for i in range(len(result) - 1):
            sum += result[i][2] / (i + 1)
            div += 1 / (i + 1)
        if div > 0:
            sum /= div
        else:
            sum = result[0][2]
        od_wait[od] = sum
    return od_length, od_wait

def process_cs(charge_od, cs, num_cs, od_length):
    cs_for_choice = []
    for od in charge_od:
        o, d = od
        cs.sort(key=lambda x: od_length[(o, x)] + od_length[(x, d)])
        cs_for_choice.append(cs[:num_cs])
    return cs_for_choice

def initialize(num_v, num_population, num_cs):
    bit_per_vehicle = math.ceil(math.log2(num_cs))
    population = []
    for i in range(num_population):
        sol = []
        for j in range(num_v * bit_per_vehicle):
            cs = random.randint(0, 1)
            sol.append(cs)
        population.append(sol)
    return population

def f1(sol, bit_per_vehicle, cs_for_choice, od_length, charge_od, beta=1):
    cost = 0
    for i in range(int(len(sol) / bit_per_vehicle)):
        cs_index = 0
        for j in range(bit_per_vehicle):
            cs_index += sol[i * bit_per_vehicle + j] * 2 ** j
        cs_id = cs_for_choice[i][cs_index]
        o, d = charge_od[i]
        cost += od_length[(o, cs_id)] + od_length[(cs_id, d)] * beta
    return cost

def f2(sol, bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od):
    lmp = {}
    bus_node = {}
    for id in cs_bus:
        lmp[id] = lmp_dict[id]
    for i in range(len(cs)):
        bus_node[cs[i]] = cs_bus[i]
    sum_value = sum(lmp.values())
    for key in lmp:
        lmp[key] /= sum_value

    cost = 0
    for i in range(int(len(sol) / bit_per_vehicle)):
        v_id = charge_v[i]
        vehicle = center.vehicles[v_id]
        cs_index = 0
        for j in range(bit_per_vehicle):
            cs_index += sol[i * bit_per_vehicle + j] * 2 ** j
        cs_id = cs_for_choice[i][cs_index]
        o, d = charge_od[i]
        if cs_id == o:
            cost += lmp[bus_node[cs_id]] * (vehicle.Emax - vehicle.E)
        else:
            cost += lmp[bus_node[cs_id]] * (vehicle.Emax - vehicle.E - od_length[(o, cs_id)] * vehicle.Edrive - od_wait[(o, cs_id)] * vehicle.Ewait)
    return cost

def single_point_crossover(parent1, parent2, bit_per_gene):
    num_gene = len(parent1) / bit_per_gene
    point = random.randint(1, num_gene - 1) * bit_per_gene
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def bitwise_mutation(sol):
    for i in range(len(sol)):
        if random.random() < 1 / len(sol):
            sol[i] = 1 - sol[i]
    return sol

def compare_individuals(args):
    i, j, fit_1, fit_2 = args
    sol_f1 = fit_1[i]
    sol_f2 = fit_2[i]
    cmp_f1 = fit_1[j]
    cmp_f2 = fit_2[j]

    if (sol_f1 < cmp_f1 and sol_f2 < cmp_f2) or (sol_f1 <= cmp_f1 and sol_f2 < cmp_f2) or (sol_f1 < cmp_f1 and sol_f2 <= cmp_f2):
        print(r"{} dominates {}".format(i, j))
        return (i, j)
    elif (sol_f1 > cmp_f1 and sol_f2 > cmp_f2) or (sol_f1 >= cmp_f1 and sol_f2 > cmp_f2) or (sol_f1 > cmp_f1 and sol_f2 >= cmp_f2):
        print(r"{} dominates {}".format(j, i))
        return (j, i)
    print(r"None dominates {} and {}".format(i, j))
    return None

def fast_non_dominated_sorting(population, bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_od, charge_v, cs_bus, lmp_dict, center, cs, processes=6):
    num = len(population)
    S = [[] for _ in range(num)]
    n = [0] * num
    rank = [0] * num
    front = [[]]
    fit_1 = []
    fit_2 = []
    for i in range(num):
        fit_1.append(f1(population[i], bit_per_vehicle, cs_for_choice, od_length, charge_od))
        fit_2.append(f2(population[i], bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od))

    with mp.Pool(processes=processes) as pool:
        args = [(i, j, fit_1, fit_2) for i in range(num - 1) for j in range(i + 1, num)]
        results = pool.map(compare_individuals, args)

    for result in results:
        if result:
            i, j = result
            S[i].append(j)
            n[j] += 1

    for i in range(num):
        if n[i] == 0:
            front[0].append(i)
            rank[i] = 1

    f = 0
    while front[f]:
        Q = []
        for i in front[f]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    Q.append(j)
                    rank[j] = f + 1
        f += 1
        front.append(Q)

    return front

def crowding_distance_assignment(population, bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_od, charge_v, cs_bus, lmp_dict, center, cs):
    num = len(population)
    distance = [0] * (num + 1)

    sol1 = population.copy()
    sol1 = sorted(sol1, key=lambda x: f1(x, bit_per_vehicle, cs_for_choice, od_length, charge_od))
    distance[0] = distance[num - 1] = math.inf
    for i in range(1, num - 1):
        distance[i] = (distance[i] + (f1(sol1[i + 1], bit_per_vehicle, cs_for_choice, od_length, charge_od) - f1(sol1[i - 1], bit_per_vehicle, cs_for_choice, od_length, charge_od))
                                        / (f1(sol1[num - 1], bit_per_vehicle, cs_for_choice, od_length, charge_od) - f1(sol1[0], bit_per_vehicle, cs_for_choice, od_length, charge_od)))

    sol2 = population.copy()
    sol2 = sorted(sol2, key=lambda x: f2(x, bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od))
    distance[0] = distance[num - 1] = math.inf
    for i in range(1, num - 1):
        distance[i] = (distance[i] + (f2(sol2[i + 1], bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od) - f2(sol2[i - 1], bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od))
                       / (f2(sol2[num - 1], bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od) - f2(sol2[0], bit_per_vehicle, cs_for_choice, od_length, od_wait, charge_v, cs_bus, lmp_dict, center, cs, charge_od)))

    return distance

def dispatch_cs_nsga2(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter):
    od_length, od_wait = process_od_length(real_path_results)
    cs_for_choice = process_cs(charge_od, cs, num_cs, od_length)
    population = initialize(len(charge_v), num_population, num_cs)

    for i in range(max_iter):
        print("第", i+1, "代")
        for j in range(0, int(len(population) / 2)):
            parent1 = population[j * 2]
            parent2 = population[j * 2 + 1]
            child1, child2 = single_point_crossover(parent1, parent2, math.ceil(math.log2(num_cs)))
            population.append(child1)
            population.append(child2)
        for j in range(len(population)):
            if random.random() < 2 / len(population):
                population[j] = bitwise_mutation(population[j])
        front = fast_non_dominated_sorting(population, math.ceil(math.log2(num_cs)), cs_for_choice, od_length, od_wait, charge_od, charge_v, cs_bus, lmp_dict, center, cs)
        last_front_population = []
        new_population = []
        len_cnt = 0
        if i < max_iter - 1:
            for kk in range(0, len(front)):
                if len_cnt == num_population:
                    break
                len_cnt += len(front[kk])
                if len_cnt > num_population:
                    for id in front[kk]:
                        last_front_population.append(population[id])
                    break
                else:
                    for id in front[kk]:
                        new_population.append(population[id])
            if last_front_population:
                ranking = crowding_distance_assignment(last_front_population, math.ceil(math.log2(num_cs)), cs_for_choice, od_length, od_wait, charge_od, charge_v, cs_bus, lmp_dict, center, cs)
                last_front_population = sorted(last_front_population, key=lambda x: ranking[last_front_population.index(x)])
                new_population += last_front_population[:num_population - len(new_population)]
            population = new_population
        else:
            final_choice = front[0]
            for id in final_choice:
                new_population.append(population[id])
            new_population = sorted(new_population, key=lambda x: f1(x, math.ceil(math.log2(num_cs)), cs_for_choice, od_length, charge_od))
            population = new_population

    best_solution = population[-1]
    bit_per_vehicle = math.ceil(math.log2(num_cs))
    real_cs_ids = []
    for i in range(len(charge_v)):
        cs_index = 0
        for j in range(bit_per_vehicle):
            cs_index += best_solution[i * bit_per_vehicle + j] * 2 ** j
        real_cs_ids.append(cs_for_choice[i][cs_index])

    return population[0], cs_for_choice, real_cs_ids










def dispatch_path_ga(solution, cs_for_choice, center, real_path_results, charge_v, charge_od, num_population, num_cs, max_iter, k = 2):

    def process_cs():
        dispatched_cs = {}
        bit_per_v = int(len(solution) / len(charge_v))
        for i in range(len(charge_v)):
            cs_index = 0
            for j in range(bit_per_v):
                cs_index += solution[i * bit_per_v + j] * 2 ** j
            cs_id = cs_for_choice[i][cs_index]
            dispatched_cs[i] = cs_id
        return dispatched_cs

    def process_edge_list():
        edge_list = {}
        for edge in center.edges.values():
            edge_list[(edge.origin, edge.destination)] = edge.id
        return edge_list

    def single_point_crossover(parent1, parent2, bit_per_gene):
        num_gene = len(parent1) / bit_per_gene
        point = random.randint(1, num_gene - 1) * bit_per_gene
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def bitwise_mutation(sol):
        for i in range(len(sol)):
            if random.random() < 1 / len(sol):
                sol[i] = 1 - sol[i]
        return sol

    def initialize(num_v):
        bit_per_vehicle = math.ceil(math.log2(k)) * 2
        population = []
        for i in range(num_population):
            sol = []
            for j in range(num_v * bit_per_vehicle):
                cs = random.randint(0, 1)
                sol.append(cs)
            population.append(sol)
        return population

    def evaluate(sol, dispatched_cs, edge_list, num_path, beta = 0.8):

        bit_per_vehicle = math.ceil(math.log2(num_path)) * 2
        # path = [[]] * len(charge_v)
        path_cnt = {}
        node_cnt = [{} for _ in range(78)]
        cost = 0
        for i in range(len(charge_v)):
            v_sol = sol[i * bit_per_vehicle: (i + 1) * bit_per_vehicle]
            v_sol_1 = v_sol[:bit_per_vehicle // 2]
            v_sol_2 = v_sol[bit_per_vehicle // 2:]
            sol_index_1 = 0
            sol_index_2 = 0
            for j in range(len(v_sol_1)):
                sol_index_1 += v_sol_1[j] * 2 ** j
                sol_index_2 += v_sol_2[j] * 2 ** j
            cs_id = dispatched_cs[i]
            # print("上一次报错")
            # print(i, len(charge_v))
            # print((charge_od[i][0], cs_id, charge_od[i][1]))
            # print(sol_index_1, sol_index_2)
            # print(real_path_results[(charge_od[i][0], cs_id)])
            # print(real_path_results[(cs_id, charge_od[i][1])])
            path1 = real_path_results[(charge_od[i][0], cs_id)][sol_index_1][0]
            path2 = real_path_results[(cs_id, charge_od[i][1])][sol_index_2][0]
            # print(print(i, len(charge_v)))
            # print("path1", path1)
            # print("path2", path2)

            cnt = 1
            for j in range(1, len(path1)):
                if (path1[j - 1], path1[j]) in path_cnt.keys():
                    path_cnt[(path1[j - 1], path1[j])] += cnt
                else:
                    path_cnt[(path1[j - 1], path1[j])] = cnt
                if j < len(path1) - 1:
                    if (path1[j - 1], path1[j + 1]) in node_cnt[path1[j]].keys():
                        node_cnt[path1[j]][(path1[j - 1], path1[j + 1])] += cnt
                    else:
                        node_cnt[path1[j]][(path1[j - 1], path1[j + 1])] = cnt
                cnt *= beta

            for j in range(1, len(path2)):
                if (path2[j - 1], path2[j]) in path_cnt.keys():
                    path_cnt[(path2[j - 1], path2[j])] += cnt
                else:
                    path_cnt[(path2[j - 1], path2[j])] = cnt
                if j < len(path2) - 1:
                    if (path2[j - 1], path2[j + 1]) in node_cnt[path2[j]].keys():
                        node_cnt[path2[j]][(path2[j - 1], path2[j + 1])] += cnt
                    else:
                        node_cnt[path2[j]][(path2[j - 1], path2[j + 1])] = cnt
                cnt *= beta

        for (o, d) in path_cnt.keys():
            edge_id = edge_list[(o, d)]
            edge = center.edges[edge_id]
            cost += path_cnt[(o, d)] * edge.capacity['all'][1] * edge.b * edge.power * (edge.k ** (edge.power - 1))

        # print("node_cnt[0]", node_cnt[0])
        # print("node_cnt[1]", node_cnt[1])
        for i in range(1, len(node_cnt)):
            if node_cnt[i] == {}:
                continue
            else:
                node = center.nodes[i]
                for (o, d) in node_cnt[i].keys():
                    fr = edge_list[(o, i)]
                    to = edge_list[(i, d)]
                    edge = center.edges[fr]
                    load = edge.capacity[to][1]
                    if load == 0: load += 1
                    k = (node_cnt[i][(o, d)] / load + 1) ** (1 / (edge.power + 1))
                    cost += (node.calculate_wait(fr, to, node.ratio[(fr, to)] * k) - node.calculate_wait(fr, to)) * edge.k

        return cost

    dispatched_cs = process_cs()
    edge_list = process_edge_list()
    population = initialize(len(charge_v))
    best = [[], float('inf')]
    for i in range(max_iter):
        for j in range(0, int(len(population) / 2)):
            parent1 = population[j * 2]
            parent2 = population[j * 2 + 1]
            child1, child2 = single_point_crossover(parent1, parent2, math.ceil(math.log2(num_cs)))
            population.append(child1)
            population.append(child2)
        for j in range(len(population)):
            if random.random() < 2 / len(population):
                population[j] = bitwise_mutation(population[j])
        new_population = sorted(population, key=lambda x: evaluate(x, dispatched_cs, edge_list, k))
        population = new_population[:num_population]
        if evaluate(population[0], dispatched_cs, edge_list, k) < best[1]:
            best = (population[0], evaluate(population[0], dispatched_cs, edge_list, k))
            print("best changed to", best[1])

    best_solution = best[0]
    bit_per_vehicle = math.ceil(math.log2(k)) * 2
    actual_paths = []
    for i in range(len(charge_v)):
        v_sol = best_solution[i * bit_per_vehicle: (i + 1) * bit_per_vehicle]
        v_sol_1 = v_sol[:bit_per_vehicle // 2]
        v_sol_2 = v_sol[bit_per_vehicle // 2:]
        sol_index_1 = 0
        sol_index_2 = 0
        for j in range(len(v_sol_1)):
            sol_index_1 += v_sol_1[j] * 2 ** j
            sol_index_2 += v_sol_2[j] * 2 ** j
        cs_id = dispatched_cs[i]
        path1 = real_path_results[(charge_od[i][0], cs_id)][sol_index_1][0]
        path2 = real_path_results[(cs_id, charge_od[i][1])][sol_index_2][0]
        if not path1:
            actual_paths.append(path2)
        else:
            actual_paths.append(path1 + path2[1:])

    return best[0], dispatched_cs, actual_paths



def generate_shortest_actual_paths(solution, charge_od, real_path_results, cs_for_choice, charge_v):
    def process_cs():
        dispatched_cs = {}
        bit_per_v = int(len(solution) / len(charge_v))
        for i in range(len(charge_v)):
            cs_index = 0
            for j in range(bit_per_v):
                cs_index += solution[i * bit_per_v + j] * 2 ** j
            cs_id = cs_for_choice[i][cs_index]
            dispatched_cs[i] = cs_id
        return dispatched_cs

    dispatched_cs = process_cs()
    actual_paths = []
    for i in range(len(charge_v)):
        cs_id = dispatched_cs[i]
        path1 = real_path_results[(charge_od[i][0], cs_id)][0][0]
        path2 = real_path_results[(cs_id, charge_od[i][1])][0][0]
        if not path1:
            actual_paths.append(path2)
        else:
            actual_paths.append(path1 + path2[1:])
    return dispatched_cs, actual_paths





























class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    def push(self, key, value):
        # Use a tuple (value, index, key) to ensure the queue is sorted by value
        heapq.heappush(self.queue, [value, key])
        self.index += 1

    def pop(self):
        # Pop the smallest value and return the entire tuple (value, index, key)
        if self.index > 0:
            self.index -= 1
            return heapq.heappop(self.queue)
        else:
            return None

    def is_empty(self):
        return len(self.queue) == 0

    def top(self):
        # Return the smallest value without removing it
        if self.index > 0:
            return self.queue[0]
        else:
            return None


def dijkstra_with_travel_time(graph, start, end, departure_time, time_constraints):
    """
    实现带有时间限制的Dijkstra最短路径算法，不考虑等待。

    参数:
        graph: 图的邻接表表示，如 {0: {1: 2, 2: 5}, ...}，其中键是节点，值是邻居节点及到达邻居节点的时间
        start: 起点节点ID
        end: 终点节点ID
        departure_time: 出发时间点
        time_constraints: 时间点限制，如 {(3, 4): [5], (0, 1): [6, 7]}，表示在特定时间点不能在特定路径上

    返回:
        (最短时间, 最短路径列表)，如果无法到达则返回(无穷大, [])
    """
    # 优先队列，存储 (总花费时间, 当前节点, 当前时间, 已访问路径)
    pq = [(0, start, departure_time, [start])]

    # 关键修改: visited字典现在存储(节点, 到达时间)的组合，而不仅仅是节点
    visited = {}

    while pq:
        total_cost, current_node, current_time, path = heapq.heappop(pq)

        # 如果到达终点，返回总花费时间和最短路径
        if current_node == end:
            return path, total_cost

        # 关键修改: 检查(节点, 时间)组合是否已访问过，而不仅仅是节点
        # 这允许我们在不同时间到达同一节点
        key = (current_node, current_time)
        if key in visited and total_cost >= visited[key]:
            continue

        # 记录访问状态
        visited[key] = total_cost

        # 遍历当前节点的所有邻居
        for neighbor, travel_time in graph[current_node].items():
            edge = (current_node, neighbor)

            # 检查是否可以在当前时间通过这条边
            can_pass = True

            # 检查边上的每个时间点
            for t in range(current_time, current_time + travel_time):
                if edge in time_constraints and t in time_constraints[edge]:
                    can_pass = False
                    break

            if not can_pass:
                continue

            # 计算新的时间和花费
            new_time = current_time + travel_time
            new_cost = total_cost + travel_time
            new_path = path + [neighbor]

            # 添加新的状态到队列中
            heapq.heappush(pq, (new_cost, neighbor, new_time, new_path))

    # 如果无法到达终点，返回无穷大和空路径
    return None, float('inf')



def dijkstra_plus(graph, start, stopover, end, start_time, time_constraints):
    """
    支持起点、经停点和终点的路径规划，带时间窗口限制。

    参数:
    - graph: 图的邻接表表示，包含边的通行时长 {'A': {'B': travel_time, ...}, ...}
    - start: 起始节点
    - stopover: 经停点
    - end: 终止节点
    - start_time: 起始时间
    - time_constraints: 时间窗口限制，格式为 {('u', 'v'): (start_time, end_time)}

    返回:
    - 总路径和通行时长，如果没有可行路径，返回 None。
    """
    # 阶段1: 从起点到经停点
    path1, travel_time1 = dijkstra_with_travel_time(graph, start, stopover, start_time, time_constraints)
    if not path1:
        return None,  float('inf'), float('inf'), float('inf')  # 如果第一阶段不可达，则直接返回

    # 阶段2: 从经停点到终点（起始时间为到达经停点的时间）
    arrival_time_at_stopover = start_time + travel_time1
    path2, travel_time2 = dijkstra_with_travel_time(graph, stopover, end, arrival_time_at_stopover, time_constraints)
    if not path2:
        return None, float('inf'), float('inf'), float('inf')  # 如果第二阶段不可达，则直接返回

    # 合并两段路径
    total_path = path1 + path2[1:]  # 避免重复经停点
    total_travel_time = travel_time1 + travel_time2

    return total_path, total_travel_time, travel_time1, travel_time2


def dispatch_CCRP(t, center, OD_ratio, cs, charge_v, anxiety_OD_ratio=None):

    # Initialize the traffic flow table
    traffic_flow = center.edge_timely_estimated_load.copy()
    for flow_set in traffic_flow.values():
        for flow in flow_set:
            flow[0] = round(flow[0])
            flow[1] = math.ceil(flow[1])

    demand = OD_ratio.copy()
    anxiety_demand = anxiety_OD_ratio.copy()

    Graph = {}
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id
        if edge.origin not in Graph.keys():
            Graph[edge.origin] = {}
        Graph[edge.origin][edge.destination] = round(edge.calculate_time())

    time_constraints = {}
    for (O, D), flow in traffic_flow.items():
        for i in range(t, len(flow)):
            if flow[i][0] >= flow[i][1]:
                if (O, D) not in time_constraints.keys():
                    time_constraints[(O, D)] = []
                time_constraints[(O, D)].append(i)


    # Initialize the result dictionary
    dispatch_result = {}
    anxiety_result = {}

    # Dispatch the vehicles
    current_time = t
    while any(demand.values()) or any(anxiety_demand.values()):
        fastest_time = float('inf')
        wait = -1
        fastest_path = None
        dispatch_od = None
        dispatch_cs = -1
        anxiety = -2
        for (O, D), count in demand.items():
            if count == 0:
                continue
            for cs_id in cs:
                for drive_time in range(current_time, current_time + 8): #这里6是考虑到每条道路的长短随便设的
                    total_path, total_travel_time, charge_travel_time, _ = dijkstra_plus(Graph, O, cs_id, D, drive_time, time_constraints)
                    if total_travel_time + drive_time - current_time < fastest_time and charge_travel_time * 0.15 * (1 + 0.1 / 3) < 4.2:
                        fastest_time = total_travel_time + drive_time - current_time
                        fastest_path = total_path
                        dispatch_od = (O, D)
                        dispatch_cs = cs_id
                        wait = drive_time - current_time
                        anxiety = 1
                        # print(fastest_time, fastest_path, dispatch_od, dispatch_cs, wait, anxiety)

        for (O, D), count in anxiety_demand.items():
            if count == 0:
                continue
            for cs_id in cs:
                for drive_time in range(current_time, current_time + 8):
                    total_path, total_travel_time, _, _ = dijkstra_plus(Graph, O, D, cs_id, drive_time, time_constraints)
                    if total_travel_time + drive_time - current_time < fastest_time and total_travel_time * 0.15 * (1 + 0.1 / 3) < 6:
                        fastest_time = total_travel_time + drive_time - current_time
                        fastest_path = total_path
                        dispatch_od = (O, D)
                        dispatch_cs = cs_id
                        wait = drive_time - current_time
                        anxiety = 0
                        # print(fastest_time, fastest_path, dispatch_od, dispatch_cs, wait, anxiety)

        if fastest_path:
            min_cap = float('inf')
            occupancy = []
            time_stamp = wait + current_time
            for p in range(len(fastest_path) - 1):
                edge_o = fastest_path[p]
                edge_d = fastest_path[p + 1]
                time_interval = Graph[edge_o][edge_d]
                for tt in range(time_stamp, min(time_stamp + time_interval, 60)):
                    cap = traffic_flow[(edge_o, edge_d)][tt][1] - traffic_flow[(edge_o, edge_d)][tt][0]
                    if cap < 0:
                        print(f"这他妈是负数？ (edge_o, edge_d): {(edge_o, edge_d)}, tt: {tt}, time_stamp: {tt}, cap: {cap},current_time: {current_time}")
                        print(f"traffic_flow: {traffic_flow}")
                        print(f"path: {fastest_path}")
                        print(f"time_constraints: {time_constraints}")
                    if cap == 0:
                        print(f"这他妈是0？ (edge_o, edge_d): {(edge_o, edge_d)}, tt: {tt}, time_stamp: {tt}, cap: {cap},current_time: {current_time}")
                        print(f"traffic_flow: {traffic_flow}")
                        print(f"path: {fastest_path}")
                        print(f"time_constraints: {time_constraints}")
                        print(f"Graph: {Graph}")
                    if cap < min_cap:
                        occupancy.clear()
                    if cap <= min_cap:
                        min_cap = cap
                        occupancy.append([(edge_o, edge_d), tt])

                time_stamp += time_interval
                if time_stamp > 60: break

            current = current_time
            if anxiety == 1:
                min_cap = min(min_cap, demand[dispatch_od])
                demand[dispatch_od] -= min_cap
            else:
                min_cap = min(min_cap, anxiety_demand[dispatch_od])
                anxiety_demand[dispatch_od] -= min_cap

            for p in range(len(fastest_path) - 1):
                edge_o = fastest_path[p]
                edge_d = fastest_path[p + 1]
                for tttt in range(current, min(current + Graph[edge_o][edge_d], 60)):
                    traffic_flow[(edge_o, edge_d)][tttt][0] += min_cap
                current += Graph[edge_o][edge_d]
                if current > 60: break

            for [od, ttt] in occupancy:
                if od not in time_constraints.keys():
                    time_constraints[od] = []
                if ttt not in time_constraints[od]:
                    time_constraints[od].append(ttt)

            if anxiety == 1:
                if current_time + wait not in dispatch_result.keys():
                    dispatch_result[current_time + wait] = {}
                dispatch_result[current_time + wait][dispatch_od] = (min_cap, dispatch_cs, fastest_path)
            else:
                if current_time + wait not in anxiety_result.keys():
                    anxiety_result[current_time + wait] = {}
                anxiety_result[current_time + wait][dispatch_od] = (min_cap, dispatch_cs, fastest_path)

            print(demand)
            print(anxiety_demand)
            print(f"dispatch {dispatch_od} to {dispatch_cs} with {min_cap} vehicles at {current_time + wait}， path{fastest_path}, anxiety{anxiety}")

        else:
            current_time += 8
            print(f"No path, current_time: {current_time}")

    return dispatch_result, traffic_flow, anxiety_result


def dispatch_CCRPP(t, center, OD_ratio, cs, charge_v, anxiety_OD_ratio=None):

    def dispatch_flow(inform, current_time):
        [O, D, cs_id, path, wait, anxiety] = inform
        min_cap = float('inf')
        occupancy = []
        time_stamp = wait + current_time
        print(traffic_flow)
        for p in range(len(path) - 1):
            edge_o = path[p]
            edge_d = path[p + 1]
            time_interval = Graph[edge_o][edge_d]
            for tt in range(time_stamp, min(time_stamp + time_interval, 60)):
                cap = traffic_flow[(edge_o, edge_d)][tt][1] - traffic_flow[(edge_o, edge_d)][tt][0]
                if cap < 0:
                    print(f"这他妈是负数？ (edge_o, edge_d): {(edge_o, edge_d)}, tt: {tt}, cap: {cap}")
                    print(f"traffic_flow: {traffic_flow}")
                    print(f"path: {path}")
                    print(f"inform: {inform}")
                    print(f"time_constraints: {time_constraints}")
                if cap == 0:
                    print(f"这他妈是0？ (edge_o, edge_d): {(edge_o, edge_d)}, tt: {tt}, cap: {cap}")
                    print(f"traffic_flow: {traffic_flow}")
                    print(f"path: {path}")
                    print(f"inform: {inform}")
                    print(f"time_constraints: {time_constraints}")
                    print(f"Graph: {Graph}")
                if cap < min_cap:
                    occupancy.clear()
                if cap <= min_cap:
                    min_cap = cap
                    occupancy.append([(edge_o, edge_d), tt])

            time_stamp += time_interval
            if time_stamp > 60: break

        current = current_time
        if anxiety == 1:
            min_cap = min(min_cap, demand[(O, D)])
            demand[(O, D)] -= min_cap

            if current_time + wait not in dispatch_result.keys():
                dispatch_result[current_time + wait] = {}
            dispatch_result[current_time + wait][(O, D)] = (min_cap, cs_id, path)
        else:
            min_cap = min(min_cap, anxiety_demand[(O, D)])
            anxiety_demand[(O, D)] -= min_cap

            if current_time + wait not in anxiety_result.keys():
                anxiety_result[current_time + wait] = {}
            anxiety_result[current_time + wait][(O, D)] = (min_cap, cs_id, path)

        for p in range(len(path) - 1):
            edge_o = path[p]
            edge_d = path[p + 1]
            for time in range(current, min(current + Graph[edge_o][edge_d], 60)):
                traffic_flow[(edge_o, edge_d)][current][0] += min_cap
            current += Graph[edge_o][edge_d]

        for [od, ttt] in occupancy:
            if od not in time_constraints.keys():
                time_constraints[od] = []
            if ttt not in time_constraints[od]:
                time_constraints[od].append(ttt)



        print(f"分配了{(O, D)}的{min_cap}辆车于{current_time + wait}, 路径为{path}")
        print(f"demand:{demand}")

        
    def check_path(inform, time_constraints, check_current = t, log = False):
        [check_O, check_D, check_cs_id, cc_path, check_wait, check_anxiety] = inform[1]
        if log:
            print(f"inform: {inform}")
            print(f"使用的参数为{check_O, check_D, check_cs_id, check_current + check_wait, time_constraints}")

        if check_anxiety == 1:
            c_path, check_time = dijkstra_plus(Graph, check_O, check_cs_id, check_D, check_current + check_wait, time_constraints)
        else:
            c_path, check_time = dijkstra_plus(Graph, check_O, check_D, check_cs_id, check_current + check_wait, time_constraints)

        if log:
            print(f"check_path: {c_path}, check_time: {check_time}")
        if check_time + check_wait == inform[0]:
            return True
        else:
            return False

    def update_path(inform, time_constraints, update_current = t, log = False):
        [update_O, update_D, u_cs_id, u_path, u_wait, update_anxiety] = inform[1]
        print(f"updating {update_O, update_D}")
        update_fastest_time = float('inf')
        update_dispatch_od = (update_O, update_D)
        update_dispatch_cs = None
        update_fastest_path = None
        update_inform = None
        update_wait = -1
        for update_cs_id in cs:
            for update_drive_time in range(update_current, update_current + 15):
                update_total_path, update_total_travel_time = dijkstra_plus(Graph, update_O, update_cs_id, update_D, update_drive_time, time_constraints)
                if update_total_travel_time + update_drive_time - update_current < update_fastest_time:
                    update_fastest_time = update_total_travel_time + update_drive_time - update_current
                    update_fastest_path = update_total_path
                    update_dispatch_cs = update_cs_id
                    update_wait = update_drive_time - update_current
                    update_inform = [update_O, update_D, update_cs_id, update_drive_time, time_constraints]
        if update_fastest_path:
            if log:
                print(f"最短路使用的参数为{update_inform}， 最早到达时间为{update_fastest_time}")
            return update_fastest_time, [update_dispatch_od[0], update_dispatch_od[1], update_dispatch_cs, update_fastest_path, update_wait]
        else:
            return None, None

    traffic_flow = center.edge_timely_estimated_load.copy()
    for flow_set in traffic_flow.values():
        for flow in flow_set:
            flow[0] = round(flow[0])
            flow[1] = math.ceil(flow[1])

    demand = OD_ratio.copy()
    anxiety_demand = anxiety_OD_ratio.copy()

    Graph = {}
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id
        if edge.origin not in Graph.keys():
            Graph[edge.origin] = {}
        Graph[edge.origin][edge.destination] = round(edge.calculate_time())

    time_constraints = {}
    for (O, D), flow in traffic_flow.items():
        for i in range(t, len(flow)):
            if flow[i][0] >= flow[i][1]:
                if (O, D) not in time_constraints.keys():
                    time_constraints[(O, D)] = []
                time_constraints[(O, D)].append(i)

    # Initialize the result dictionary
    dispatch_result = {}
    anxiety_result = {}

    RQ = PriorityQueue()
    Pre_RQ = PriorityQueue()

    current_time = t
    for (O, D), count in demand.items():
        fastest_time = float('inf')
        fastest_path = None
        fastest_cs = -1
        if count == 0:
            continue
        for cs_id in cs:
            total_path, total_travel_time, charge_drive_time, _ = dijkstra_plus(Graph, O, cs_id, D, current_time, {})
            if total_path and total_travel_time < fastest_time and charge_drive_time * 0.15 * (1 + 0.1 / 3) < 4.2:
                fastest_time = total_travel_time
                fastest_path = total_path
                fastest_cs = cs_id
        if fastest_path:
            Pre_RQ.push([O, D, fastest_cs, fastest_path, 0, 1], fastest_time)

    for (O, D), count in anxiety_demand.items():
        fastest_time = float('inf')
        fastest_path = None
        fastest_cs = -1
        if count == 0:
            continue
        for cs_id in cs:
            total_path, total_travel_time, _, _ = dijkstra_plus(Graph, O, D, cs_id, current_time, {})
            if total_path and total_travel_time < fastest_time and total_travel_time * 0.15 * (1 + 0.1 / 3) < 6:
                fastest_time = total_travel_time
                fastest_path = total_path
                fastest_cs = cs_id
        if fastest_path:
            Pre_RQ.push([O, D, fastest_cs, fastest_path, 0, 0], fastest_time)

    (k, v) = Pre_RQ.pop()
    RQ.push(v, k)

    main_loop_cnt = 0
    while any(demand.values()):
        main_loop_cnt += 1
        print(f"main_loop: {main_loop_cnt}")

        for (O, D), flow in traffic_flow.items():
            for ii in range(t, len(flow)):
                if flow[ii][0] >= flow[ii][1]:
                    if (O, D) not in time_constraints.keys():
                        time_constraints[(O, D)] = []
                    if ii not in time_constraints[(O, D)]:
                        time_constraints[(O, D)].append(ii)
        print(f"time_constraints: {time_constraints}")

        while RQ.top == [None, None]:
            RQ.pop()

        while RQ.top() is None:
            PP = Pre_RQ.pop()
            print(f"RQ空着", end=' ')
            if check_path(PP, time_constraints):
                RQ.push(PP[1], PP[0])
                print(f"RQ有了{PP}")
            else:
                new_path, new_inform = update_path(PP, time_constraints)
                Pre_RQ.push(new_inform, new_path)
                print(f"原来是{PP}, 更新了{[new_path, new_inform]}回到Pre_RQ")
        Q1 = RQ.top()
        P1 = Pre_RQ.top()
        print(f"Q1: {Q1}, P1: {P1}")
        if not P1:
            P1 = [float('inf'), [-1, -1, -1, [], -1]]
        if P1[0] < Q1[0]:
            Pre_RQ.pop()
            if check_path(P1, time_constraints):
                RQ.push(P1[1], P1[0])
            else:
                new_path, new_inform = update_path(P1, time_constraints)
                Pre_RQ.push(new_inform, new_path)

        else:
            RQ.pop()
            O = Q1[1][0]
            D = Q1[1][1]


            Q2 = RQ.top()
            PQ = Q2 if Q2 else P1


            print(f"Q1: {Q1}, Q2: {Q2}, PQ: {PQ}")
            loop_cnt = 0
            while Q1[0] <= PQ[0] and demand[(O, D)] > 0:
                loop_cnt += 1
                print(f"loop_cnt: {loop_cnt}")
                if check_path(Q1, time_constraints):
                    dispatch_flow(Q1[1], current_time)
                if demand[(O, D)] > 0:
                    new_Q1_path, new_Q1_inform = update_path(Q1, time_constraints)
                    Q1 = [new_Q1_path, new_Q1_inform]
                    print(f"Q1: {Q1}, Q2: {Q2}, PQ: {PQ}")
                elif demand[(O, D)] == 0:
                    break

            if demand[(O, D)] > 0:
                if Q2:
                    Pre_RQ.push(Q1[1], Q1[0])
                else:
                    RQ.push(Q1[1], Q1[0])

    return dispatch_result, traffic_flow


def update_center_for_heuristic(center, dispatch_result, current_time, charge_v, anxiety_result=None):
    #dispatch_result[current_time + wait][(O, D)] = (min_cap, cs_id, path)

    vehicle_ids = {}
    anxiety_vehicle_ids = {}
    index = 0
    for vehicle_id in charge_v:
        vehicle = center.vehicles[vehicle_id]
        O = vehicle.origin
        D = vehicle.destination
        if vehicle.anxiety == 1:
            if (O, D) not in vehicle_ids.keys():
                vehicle_ids[(O, D)] = []
            vehicle_ids[(O, D)].append(vehicle_id)
        else:
            if (O, D) not in anxiety_vehicle_ids.keys():
                anxiety_vehicle_ids[(O, D)] = []
            anxiety_vehicle_ids[(O, D)].append(vehicle_id)



    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id

    for drive_time in dispatch_result.keys():
        for (O, D), (min_cap, cs_id, path) in dispatch_result[drive_time].items():
            cnt = 0
            while cnt < min_cap and vehicle_ids[(O, D)]:
                vehicle_id = vehicle_ids[(O, D)].pop(0)
                vehicle = center.vehicles[vehicle_id]
                vehicle.path = []
                for i in range(len(path) - 1):
                    vehicle.path.append(edge_od_id[(path[i], path[i + 1])])
                vehicle.road = vehicle.path[0]
                vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1
                vehicle.distance = center.edges[vehicle.road].length
                vehicle.speed = center.edges[vehicle.road].calculate_drive()
                vehicle.charge = (cs_id, 300)

                flow_ind = drive_time
                for path_ind in range(0, len(vehicle.path)):
                    edge_id = vehicle.path[path_ind]
                    edge_o = center.edges[edge_id].origin
                    edge_d = center.edges[edge_id].destination
                    time_interval = round(center.edges[edge_id].calculate_time())
                    # print(f"edge_id: {edge_id}, edge_o: {edge_o}, edge_d: {edge_d}, time_interval: {time_interval}")
                    while time_interval >= 1 and flow_ind <= 60:
                        center.edge_timely_estimated_load[(edge_o, edge_d)][flow_ind][0] += 1
                        time_interval -= 1
                        flow_ind += 1

                if drive_time == current_time:
                    center.edges[vehicle.road].capacity["all"] = center.solve_tuple(center.edges[vehicle.road].capacity["all"], 1)
                    center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(center.edges[vehicle.road].capacity["charge"], 1)
                    center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(center.edges[vehicle.road].capacity[vehicle.next_road], 1)
                    vehicle.drive()
                else:
                    vehicle.delay = True
                    center.delay_vehicles[drive_time].append(vehicle.id)

    for anxiety_drive_time in anxiety_result.keys():
        for (O, D), (min_cap, cs_id, path) in anxiety_result[anxiety_drive_time].items():
            cnt = 0
            while cnt < min_cap and anxiety_vehicle_ids[(O, D)]:
                vehicle_id = anxiety_vehicle_ids[(O, D)].pop(0)
                vehicle = center.vehicles[vehicle_id]
                vehicle.path = []
                for i in range(len(path) - 1):
                    vehicle.path.append(edge_od_id[(path[i], path[i + 1])])
                vehicle.road = vehicle.path[0]
                vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1
                vehicle.distance = center.edges[vehicle.road].length
                vehicle.speed = center.edges[vehicle.road].calculate_drive()
                vehicle.charge = (cs_id, 300)
                vehicle.destination = cs_id

                flow_ind = anxiety_drive_time
                for path_ind in range(0, len(vehicle.path)):
                    edge_id = vehicle.path[path_ind]
                    edge_o = center.edges[edge_id].origin
                    edge_d = center.edges[edge_id].destination
                    time_interval = round(center.edges[edge_id].calculate_time())
                    # print(f"edge_id: {edge_id}, edge_o: {edge_o}, edge_d: {edge_d}, time_interval: {time_interval}")
                    while time_interval >= 1 and flow_ind <= 60:
                        center.edge_timely_estimated_load[(edge_o, edge_d)][flow_ind][0] += 1
                        time_interval -= 1
                        flow_ind += 1

                if anxiety_drive_time == current_time:
                    center.edges[vehicle.road].capacity["all"] = center.solve_tuple(center.edges[vehicle.road].capacity["all"], 1)
                    center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(center.edges[vehicle.road].capacity["charge"], 1)
                    center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(center.edges[vehicle.road].capacity[vehicle.next_road], 1)
                    vehicle.drive()
                else:
                    vehicle.delay = True
                    center.delay_vehicles[anxiety_drive_time].append(vehicle.id)


def dispatch_CASPER(center):
    Graph = {}
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id
        if edge.origin not in Graph.keys():
            Graph[edge.origin] = {}
        Graph[edge.origin][edge.destination] = {"cap": edge.capacity, "imp": edge.calculate_time()}

    return None