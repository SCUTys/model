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


##MOPSO
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

def process_cs(OD_ratio, cs, num_cs, od_length, anxiety = 1):
    cs_for_choice = {}
    for od in OD_ratio.keys():
        o, d = od
        if anxiety == 1:
            cs.sort(key=lambda x: od_length[(o, x)] + od_length[(x, d)])
        else:
            cs.sort(key=lambda x: od_length[(x, d)])
        cs_for_choice[od] = cs[:num_cs]
    return cs_for_choice

def generate_random_numbers(n):
    random_numbers = np.random.uniform(0, 1, n)
    random_numbers /= random_numbers.sum()
    random_numbers = list(random_numbers)
    return random_numbers

def initialize_population(num_population, num_cs, od_num):
    population = []
    velocity = []
    for i in range(num_population):
        individual = []
        velocity.append([])
        for j in range(od_num):
            individual.append(generate_random_numbers(num_cs))
            velocity[-1].append([0] * num_cs)
        population.append(individual)
    return population, velocity

def mutation(individual, mutation_rate):

    return individual

def f1(individual, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice):
    f1_value = 0
    od_len_f1 = len(OD_ratio.keys())
    cs_choice_cnt = len(individual[0])
    for i in range(od_len_f1 * 2):
        if i < od_len_f1:
            od = list(OD_ratio.keys())[i]
            o, d = od
            for j in range(cs_choice_cnt):
                cs = cs_for_choice[od][j]
                f1_value += individual[i][j] * OD_ratio[od] * (od_length[(o, cs)] + od_length[(cs, d)])
        else:
            od = list(anxiety_OD_ratio.keys())[i - od_len_f1]
            o, d = od
            for j in range(cs_choice_cnt):
                cs = anxiety_cs_for_choice[od][j]
                f1_value += individual[i][j] * anxiety_OD_ratio[od] * (od_length[(o, d)] + od_length[(d, cs)])

    return f1_value

def f2(individual, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait):
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
    cs_choice_cnt = len(individual[0])
    for i in range(od_len_f2 * 2):
        if i < od_len_f2:
            od = list(OD_ratio.keys())[i]
            o, d = od
            for j in range(cs_choice_cnt):
                cs_id = cs_for_choice[od][j]
                f2_value += individual[i][j] * OD_ratio[od] * lmp[bus_node[cs_id]] * (54 - 4 + od_length[(o, cs_id)] * 0.15 - od_wait[(o, cs_id)] * 0.1)
        else:
            od = list(anxiety_OD_ratio.keys())[i - od_len_f2]
            o, d = od
            for j in range(cs_choice_cnt):
                cs_id = anxiety_cs_for_choice[od][j]
                f2_value += individual[i][j] * anxiety_OD_ratio[od] * lmp[bus_node[cs_id]] * (54 - 6 + 0.15 * (od_length[(o, d)] + od_length[(d, cs_id)]) - 0.1 * (od_wait[(o, d)] + od_wait[(d, cs_id)]))

    return f2_value


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


def fast_non_dominated_sorting(population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait, cs_bus, lmp_dict, cs, processes=6):
    num = len(population)
    S = [[] for _ in range(num)]
    n = [0] * num
    rank = [0] * num
    front = [[]]
    fit_1 = []
    fit_2 = []
    for i in range(num):
        fit_1.append(f1(population[i], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice))
        fit_2.append(f2(population[i], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait))

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

    return front, fit_1, fit_2


def cube_selection(REP, n, num_population):
    if not REP:
        return None

    # 步骤1：求得REP中fit1和fit2的上下限
    fit1_values = [rep[0] for rep in REP]
    fit2_values = [rep[1] for rep in REP]

    min_fit1 = min(fit1_values) - 0.1
    max_fit1 = max(fit1_values) + 0.1
    min_fit2 = min(fit2_values) - 0.1
    max_fit2 = max(fit2_values) + 0.1

    # 避免除零错误
    if max_fit1 == min_fit1:
        max_fit1 += 1e-10
    if max_fit2 == min_fit2:
        max_fit2 += 1e-10

    # 步骤2：以求出的上下限形成一方格，并平均分割成n*n的网格
    grid = {}  # 用字典存储每个网格中的粒子

    # 步骤3：记录REP的元素分别在哪个网格中
    for idx, (fit1, fit2, particle) in enumerate(REP):
        # 计算粒子所在的网格位置
        grid_x = min(n - 1, int(n * (fit1 - min_fit1) / (max_fit1 - min_fit1)))
        grid_y = min(n - 1, int(n * (fit2 - min_fit2) / (max_fit2 - min_fit2)))

        grid_pos = (grid_x, grid_y)
        if grid_pos not in grid:
            grid[grid_pos] = []

        grid[grid_pos].append((fit1, fit2, particle, idx))  # 保存粒子索引

    # 步骤4：为每个网格赋一个适应度值
    grid_fitness = {}
    for grid_pos, particles in grid.items():
        num_particles = len(particles)
        grid_fitness[grid_pos] = num_population / 10 / num_particles

    # 步骤5：将所有适应度归一化后轮盘赌选择其中一网格，并在此网格中随机选取随机一个体
    total_fitness = sum(grid_fitness.values())
    fitness_normalized = {grid_pos: fitness / total_fitness for grid_pos, fitness in grid_fitness.items()}

    # 轮盘赌选择一个网格
    r = random.random()
    cumulative_prob = 0
    selected_grid = None

    for grid_pos, fitness in fitness_normalized.items():
        cumulative_prob += fitness
        if r <= cumulative_prob:
            selected_grid = grid_pos
            break

    # 从选中的网格中随机选择一个粒子
    selected_particles = grid[selected_grid]
    selected_particle = random.choice(selected_particles)

    # 返回选中的粒子
    return selected_particle[2]  # 返回粒子本身(不包括适应度值和索引)


def update_velocity(population, velocity, p_memory, p_REP, w):
    # print(population)
    # print(len(population), len(population[0]), len(population[0][0]))
    # print(velocity)
    # print(len(velocity), len(velocity[0]), len(velocity[0][0]))
    # print(p_REP)
    # print(len(p_REP), len(p_REP[0]))
    for i in range(len(population)):
        p_best = p_memory[i][2]
        # print(p_best)
        # print(len(p_best), len(p_best[0]))
        for j in range(len(population[i])):
            for k in range(len(population[i][j])):
                r1 = random.random()
                r2 = random.random()
                velocity[i][j][k] = w * velocity[i][j][k] + r1 * (p_best[j][k] - population[i][j][k]) + r2 * (p_REP[j][k] - population[i][j][k])
    return velocity


def update_position(population, velocity):
    for i in range(len(population)):
        for j in range(len(population[i])):
            # 更新位置：位置 = 位置 + 速度
            for k in range(len(population[i][j])):
                population[i][j][k] += velocity[i][j][k]

            # 将负数取反
            for k in range(len(population[i][j])):
                if population[i][j][k] < 0:
                    population[i][j][k] = -population[i][j][k]

            # 归一化，使和为1
            sum_value = sum(population[i][j])
            if sum_value > 0:  # 避免除以零
                for k in range(len(population[i][j])):
                    population[i][j][k] /= sum_value
            else:  # 如果所有值都是0，则平均分配
                for k in range(len(population[i][j])):
                    population[i][j][k] = 1.0 / len(population[i][j])

    return population


def update_REP(REP, population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait,
               cs_bus, lmp_dict, cs, REP_max):
    # 计算新种群中每个粒子的适应度值
    new_fits = []
    for i in range(len(population)):
        fit1_val = f1(population[i], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
        fit2_val = f2(population[i], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                      anxiety_cs_for_choice, od_length, od_wait)
        new_fits.append((fit1_val, fit2_val, population[i]))

    # 检查每个新粒子是否应该加入 REP
    for fit in new_fits:
        dominated = False
        dominated_indices = []

        # 检查新粒子是否被现有 REP 中的解支配
        # 同时找出它能支配的 REP 中的解
        for j, rep in enumerate(REP):
            if (rep[0] < fit[0] and rep[1] < fit[1]) or \
                    (rep[0] <= fit[0] and rep[1] < fit[1]) or \
                    (rep[0] < fit[0] and rep[1] <= fit[1]):
                dominated = True
                break

            if (fit[0] < rep[0] and fit[1] < rep[1]) or \
                    (fit[0] <= rep[0] and fit[1] < rep[1]) or \
                    (fit[0] < rep[0] and fit[1] <= rep[1]):
                dominated_indices.append(j)

        # 如果新粒子不被支配，应当加入 REP
        if not dominated:
            # 移除被新粒子支配的解
            dominated_indices.sort(reverse=True)
            for idx in dominated_indices:
                REP.pop(idx)

            # 添加新粒子到 REP
            REP.append(fit)

    # 如果 REP 数量超过上限，需要移除一些解
    if len(REP) > REP_max:
        # 步骤1：求 REP 中 fit1 和 fit2 的范围
        fit1_values = [rep[0] for rep in REP]
        fit2_values = [rep[1] for rep in REP]

        min_fit1 = min(fit1_values) - 0.1
        max_fit1 = max(fit1_values) + 0.1
        min_fit2 = min(fit2_values) - 0.1
        max_fit2 = max(fit2_values) + 0.1

        # 避免除零错误
        if max_fit1 == min_fit1:
            max_fit1 += 1e-10
        if max_fit2 == min_fit2:
            max_fit2 += 1e-10

        # 步骤2和3：将 REP 中的粒子分配到网格
        n = 10  # 网格的划分数
        grid = {}

        for idx, (fit1, fit2, particle) in enumerate(REP):
            grid_x = min(n - 1, int(n * (fit1 - min_fit1) / (max_fit1 - min_fit1)))
            grid_y = min(n - 1, int(n * (fit2 - min_fit2) / (max_fit2 - min_fit2)))

            grid_pos = (grid_x, grid_y)
            if grid_pos not in grid:
                grid[grid_pos] = []

            grid[grid_pos].append(idx)

        # 找出包含粒子最多的网格
        max_particles = 0
        crowded_grid = None

        for grid_pos, indices in grid.items():
            if len(indices) > max_particles:
                max_particles = len(indices)
                crowded_grid = grid_pos

        # 从包含粒子最多的网格中随机选一个粒子移除
        if crowded_grid:
            remove_idx = random.choice(grid[crowded_grid])
            REP.pop(remove_idx)

    return REP


def dispatch_cs_MOPSO(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None):
    od_length, od_wait = process_od_length(real_path_results)
    population, velocity = initialize_population(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    REP = []
    REP_max = num_population / 5
    front, fit1, fit2 = fast_non_dominated_sorting(population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait, cs_bus, lmp_dict, cs)
    print('Initialized fast non dominated sorting finished')
    for i in front[0]:
        REP.append((fit1[i], fit2[i], population[i]))
    print('Initialized REP finished')
    p_memory = []
    for i in range(num_population):
        p_memory.append((fit1[i], fit2[i], population[i]))
    print('Initialized p_memory finished')

    for loop_cnt in range(max_iter):
        print("第{}代".format(loop_cnt))
        p_REP = cube_selection(REP, 3, num_population)
        print("p_REP")
        velocity = update_velocity(population, velocity, p_memory, p_REP, 0.4)
        print("velocity updated")
        population = update_position(population, velocity)
        print("population updated")
        REP = update_REP(REP, population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice, od_length,
                         od_wait, cs_bus, lmp_dict, cs, REP_max)
        print("REP updated")

        # 更新个体最优
        for i in range(num_population):
            fit1_val = f1(population[i], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            fit2_val = f2(population[i], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                          anxiety_cs_for_choice, od_length, od_wait)

            # 如果新位置支配旧的个体最优，则更新个体最优
            if ((fit1_val < p_memory[i][0] and fit2_val < p_memory[i][1]) or
                    (fit1_val <= p_memory[i][0] and fit2_val < p_memory[i][1]) or
                    (fit1_val < p_memory[i][0] and fit2_val <= p_memory[i][1])):
                p_memory[i] = (fit1_val, fit2_val, population[i])

    REP = sorted(REP, key=lambda x: x[0])
    return REP, cs_for_choice, anxiety_cs_for_choice


def dispatch_vehicles_by_mopso(center, REP, charge_v, OD_ratio, cs_for_choice, real_path_results,
                               anxiety_OD_ratio, anxiety_cs_for_choice):
    """
    基于MOPSO算法的优化结果分配车辆到充电站，并使用轮盘赌方法决定每辆车的充电站

    Args:
        center: 调度中心
        REP: MOPSO算法的非支配解集，按fit1值升序排序
        charge_v: 需要充电的车辆ID列表
        charge_od: OD对到车辆ID的映射
        cs_for_choice: 普通车辆每个OD对可选择的充电站列表
        real_path_results: 包含最短路径信息的字典
        anxiety_cs_for_choice: 焦虑车辆每个OD对可选择的充电站列表

    Returns:
        dispatch_result: 普通车辆的分配结果字典
        anxiety_result: 焦虑车辆的分配结果字典
    """

    # 使用fit1值最小的解作为最优解
    best_solution = REP[0][2]  # 从(fit1, fit2, solution)元组中提取solution

    # 创建从节点对到边ID的映射
    edge_od_id = {}
    for edge in center.edges.values():
        edge_od_id[(edge.origin, edge.destination)] = edge.id

    # 准备结果容器
    dispatch_result = {}
    anxiety_result = {}
    current_time = 0  # 所有车辆同时出发

    # 按OD对和车辆类型分组
    normal_vehicles = {}  # 普通车辆: {od_pair: [vehicle_ids]}
    anxiety_vehicles = {}  # 焦虑车辆: {od_pair: [vehicle_ids]}

    for v_id in charge_v:
        vehicle = center.vehicles[v_id]
        od_pair = (vehicle.origin, vehicle.destination)

        if vehicle.anxiety == 1:  # 普通车辆
            if od_pair not in normal_vehicles:
                normal_vehicles[od_pair] = []
            normal_vehicles[od_pair].append(v_id)
        else:  # 焦虑车辆
            if od_pair not in anxiety_vehicles:
                anxiety_vehicles[od_pair] = []
            anxiety_vehicles[od_pair].append(v_id)

    # 处理普通车辆
    od_idx = 0
    for od in OD_ratio.keys():
        if od not in cs_for_choice:
            od_idx += 1
            continue

        # 获取该OD对的充电站概率分布
        cs_probs = best_solution[od_idx]

        # 计算累积概率，用于轮盘赌选择
        cum_probs = []
        curr_sum = 0
        for prob in cs_probs:
            curr_sum += prob
            cum_probs.append(curr_sum)

        # 为每个车辆通过轮盘赌选择充电站
        cs_assigned_vehicles = {cs_id: [] for cs_id in cs_for_choice[od]}

        for vehicle_id in normal_vehicles[od]:
            # 轮盘赌选择
            r = random.random()
            selected_cs_index = 0

            for i, threshold in enumerate(cum_probs):
                if r <= threshold:
                    selected_cs_index = i
                    break

            # 获取所选充电站
            if selected_cs_index < len(cs_for_choice[od]):
                cs_id = cs_for_choice[od][selected_cs_index]
                cs_assigned_vehicles[cs_id].append(vehicle_id)

        # 为每个充电站的车辆准备路径和分配结果
        for cs_id, vehicle_ids in cs_assigned_vehicles.items():
            if not vehicle_ids:
                continue

            # 获取实际路径从real_path_results
            # 普通车辆：起点 -> 充电站 -> 终点
            o_cs_path = real_path_results.get((od[0], cs_id), [])
            cs_d_path = real_path_results.get((cs_id, od[1]), [])
            if not o_cs_path or not cs_d_path:
                continue  # 如果没有找到路径，跳过

            # 构建完整节点路径：起点->充电站->终点
            o_cs_nodes = o_cs_path[0][0]
            cs_d_nodes = cs_d_path[0][0]

            # 避免重复节点（充电站）
            full_path = o_cs_nodes + cs_d_nodes[1:]

            # 存储到分配结果
            if current_time not in dispatch_result:
                dispatch_result[current_time] = {}

            dispatch_result[current_time][od] = (len(vehicle_ids), cs_id, full_path)

            # 更新车辆属性并开始行驶
            for vehicle_id in vehicle_ids:
                vehicle = center.vehicles[vehicle_id]

                # 设置车辆路径 - 将节点路径转换为边路径
                vehicle.path = []
                for i in range(len(full_path) - 1):
                    if (full_path[i], full_path[i + 1]) in edge_od_id:
                        vehicle.path.append(edge_od_id[(full_path[i], full_path[i + 1])])

                if not vehicle.path:
                    continue  # 如果路径为空，跳过

                # 设置初始道路和下一道路
                vehicle.road = vehicle.path[0]
                vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

                # 设置距离和速度
                vehicle.distance = center.edges[vehicle.road].length
                vehicle.speed = center.edges[vehicle.road].calculate_drive()

                # 设置充电站
                vehicle.charge = (cs_id, 300)  # 300秒充电时间

                # 更新容量计数
                center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["all"], 1)
                center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["charge"], 1)
                if vehicle.next_road != -1:
                    center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                        center.edges[vehicle.road].capacity[vehicle.next_road], 1)

                # 开始行驶
                print(f"Vehicle {vehicle_id} assigned to CS {cs_id} with path {vehicle.path}")
                vehicle.drive()

        od_idx += 1

    # 处理焦虑车辆
    if anxiety_cs_for_choice:
        for od in anxiety_OD_ratio.keys():
            if od not in anxiety_cs_for_choice:
                od_idx += 1
                continue

            # 获取该OD对的充电站概率分布
            cs_probs = best_solution[od_idx]

            # 计算累积概率，用于轮盘赌选择
            cum_probs = []
            curr_sum = 0
            for prob in cs_probs:
                curr_sum += prob
                cum_probs.append(curr_sum)

            # 为每个车辆通过轮盘赌选择充电站
            cs_assigned_vehicles = {cs_id: [] for cs_id in anxiety_cs_for_choice[od]}

            for vehicle_id in anxiety_vehicles[od]:
                # 轮盘赌选择
                r = random.random()
                selected_cs_index = 0

                for i, threshold in enumerate(cum_probs):
                    if r <= threshold:
                        selected_cs_index = i
                        break

                # 获取所选充电站
                if selected_cs_index < len(anxiety_cs_for_choice[od]):
                    cs_id = anxiety_cs_for_choice[od][selected_cs_index]
                    cs_assigned_vehicles[cs_id].append(vehicle_id)

            # 为每个充电站的车辆准备路径和分配结果
            for cs_id, vehicle_ids in cs_assigned_vehicles.items():
                if not vehicle_ids:
                    continue

                # 获取实际路径从real_path_results
                # 焦虑车辆：起点 -> 终点 -> 充电站
                o_d_path = real_path_results.get((od[0], od[1]), [])
                d_cs_path = real_path_results.get((od[1], cs_id), [])
                if not o_d_path or not d_cs_path:
                    continue  # 如果没有找到路径，跳过

                # 构建完整节点路径：起点->终点->充电站
                o_d_nodes = o_d_path[0][0]
                d_cs_nodes = d_cs_path[0][0]

                # 避免重复节点（终点）
                full_path = o_d_nodes + d_cs_nodes[1:]

                # 存储到分配结果
                if current_time not in anxiety_result:
                    anxiety_result[current_time] = {}

                anxiety_result[current_time][od] = (len(vehicle_ids), cs_id, full_path)

                # 更新车辆属性并开始行驶
                for vehicle_id in vehicle_ids:
                    vehicle = center.vehicles[vehicle_id]

                    # 设置车辆路径 - 将节点路径转换为边路径
                    vehicle.path = []
                    for i in range(len(full_path) - 1):
                        if (full_path[i], full_path[i + 1]) in edge_od_id:
                            vehicle.path.append(edge_od_id[(full_path[i], full_path[i + 1])])

                    if not vehicle.path:
                        continue  # 如果路径为空，跳过

                    # 设置初始道路和下一道路
                    vehicle.road = vehicle.path[0]
                    vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

                    # 设置距离和速度
                    vehicle.distance = center.edges[vehicle.road].length
                    vehicle.speed = center.edges[vehicle.road].calculate_drive()

                    # 设置充电站
                    vehicle.charge = (cs_id, 300)  # 300秒充电时间

                    # 更新容量计数
                    center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["all"], 1)
                    center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["charge"], 1)
                    if vehicle.next_road != -1:
                        center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                            center.edges[vehicle.road].capacity[vehicle.next_road], 1)

                    # 开始行驶
                    vehicle.drive()

            od_idx += 1

    return dispatch_result, anxiety_result