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


def mutation(population, mutation_rate, current_iter, max_iter):
    # 计算当前迭代的变异概率
    mutation_prob = (1 - current_iter / max_iter) ** (5 / mutation_rate)

    for i in range(len(population)):
        if random.random() < mutation_prob:
            # 随机选择一个子列表
            sub_list_idx = random.randint(0, len(population[i]) - 1)
            sub_list = population[i][sub_list_idx]

            # 随机选择子列表中的一项进行变异
            mutation_idx = random.randint(0, len(sub_list) - 1)

            # 确定上下界
            upper_bound = min(1, sub_list[mutation_idx] + mutation_prob)
            lower_bound = max(0, sub_list[mutation_idx] - mutation_prob)

            # 在上下界间随机生成一个新值
            new_value = random.uniform(lower_bound, upper_bound)

            # 保存原始值，用于计算调整系数
            original_value = sub_list[mutation_idx]

            # 替换为新值
            sub_list[mutation_idx] = new_value

            # 计算其他项需要调整的总量
            adjust_total = 1 - new_value

            # 计算其他项的原始总和
            other_sum = sum(sub_list) - new_value

            # 如果其他项总和为0，均匀分配
            if other_sum <= 1e-10:
                other_count = len(sub_list) - 1
                if other_count > 0:  # 防止除以零
                    for j in range(len(sub_list)):
                        if j != mutation_idx:
                            sub_list[j] = adjust_total / other_count
            else:
                # 按比例调整其他项
                scale_factor = adjust_total / other_sum
                for j in range(len(sub_list)):
                    if j != mutation_idx:
                        sub_list[j] *= scale_factor

    return population

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
        # print(r"{} dominates {}".format(i, j))
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

    min_fit1 = min(fit1_values) - 0.01 * (max(fit1_values) - min(fit1_values))
    max_fit1 = max(fit1_values) + 0.01 * (max(fit1_values) - min(fit1_values))
    min_fit2 = min(fit2_values) - 0.01 * (max(fit2_values) - min(fit2_values))
    max_fit2 = max(fit2_values) + 0.01 * (max(fit2_values) - min(fit2_values))

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

        min_fit1 = min(fit1_values) - 0.01 * (max(fit1_values) - min(fit1_values))
        max_fit1 = max(fit1_values) + 0.01 * (max(fit1_values) - min(fit1_values))
        min_fit2 = min(fit2_values) - 0.01 * (max(fit2_values) - min(fit2_values))
        max_fit2 = max(fit2_values) + 0.01 * (max(fit2_values) - min(fit2_values))

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


def cleanup_REP(REP):
    """
    清理REP中被支配的解，确保REP只包含非支配解

    Args:
        REP: 非支配解集，格式为[(fit1, fit2, individual), ...]

    Returns:
        清理后的非支配解集
    """
    if not REP or len(REP) <= 1:
        return REP

    # 复制一份REP进行处理
    cleaned_REP = REP.copy()

    # 标记需要删除的索引
    to_delete = []

    # 两两比较所有解
    for i in range(len(cleaned_REP)):
        if i in to_delete:
            continue

        fit1_i = cleaned_REP[i][0]
        fit2_i = cleaned_REP[i][1]

        for j in range(len(cleaned_REP)):
            if i == j or j in to_delete:
                continue

            fit1_j = cleaned_REP[j][0]
            fit2_j = cleaned_REP[j][1]

            # 检查j是否支配i
            if (fit1_j < fit1_i and fit2_j < fit2_i) or \
                    (fit1_j <= fit1_i and fit2_j < fit2_i) or \
                    (fit1_j < fit1_i and fit2_j <= fit2_i):
                to_delete.append(i)
                break

            # 检查i是否支配j
            if (fit1_i < fit1_j and fit2_i < fit2_j) or \
                    (fit1_i <= fit1_j and fit2_i < fit2_j) or \
                    (fit1_i < fit1_j and fit2_i <= fit2_j):
                to_delete.append(j)

    # 从大到小排序待删除索引，以避免删除后索引变化导致错误
    to_delete = sorted(set(to_delete), reverse=True)

    # 删除被标记的解
    for idx in to_delete:
        cleaned_REP.pop(idx)

    return cleaned_REP

def dispatch_cs_MOPSO(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None, P_first = 1):
    od_length, od_wait = process_od_length(real_path_results)
    population, velocity = initialize_population(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    REP = []
    REP_max = num_population
    P_s = 1
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
        population = mutation(population, 0.5, loop_cnt, max_iter)
        print("mutation finished")
        velocity = update_velocity(population, velocity, p_memory, p_REP, 0.4 + 0.5 * P_s)
        print("velocity updated")
        population = update_position(population, velocity)
        print("population updated")
        REP = update_REP(REP, population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice, od_length,
                         od_wait, cs_bus, lmp_dict, cs, REP_max)
        print("REP updated")

        u_l_cnt = 0

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
                u_l_cnt += 1
        P_s = u_l_cnt / num_population
        print("个体最优更新率：", P_s)

    REP = sorted(REP, key=lambda x: x[P_first])
    print(f"best_solution: {REP}")
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
    print("最优解：", REP)
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
                vehicle.charge = (cs_id, 300)

                # 更新容量计数
                center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["all"], 1)
                if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                    center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["charge"], 1)
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

    print(f"best_solution: {best_solution}")
    print(REP[0])
    print(REP[-1])
    return dispatch_result, anxiety_result


########################################################################################################################
#MODE

def deal_with_list(l1, l2, operation):
    """
    对两个列表或NumPy数组执行元素级的运算操作

    Args:
        l1: 第一个列表或数组
        l2: 第二个列表或数组
        operation: 运算类型，可以是 '+', '-', '*' 等

    Returns:
        包含运算结果的新列表
    """
    # print("l1:", l1)
    # print("l2:", l2)
    # 检查输入是否为空
    if isinstance(l1, list) and not l1:
        return []
    if isinstance(l2, list) and not l2:
        return []

    # 检查是否为嵌套列表/数组
    if isinstance(l1, list) and isinstance(l1[0], list):
        # 如果是嵌套列表，递归处理
        result = []
        for i in range(len(l1)):
            result.append(deal_with_list(l1[i], l2[i], operation))
        return result
    else:
        # 基本列表元素的运算
        if operation == '+':
            return [a + b for a, b in zip(l1, l2)]
        elif operation == '-':
            return [a - b for a, b in zip(l1, l2)]
        elif operation == '*':
            return [a * b for a, b in zip(l1, l2)]
        else:
            raise ValueError(f"不支持的运算: {operation}")


def initialize_MODE(num_population, num_cs, od_num):
    population = []
    for i in range(num_population):
        individual = []
        for j in range(od_num):
            individual.append(generate_random_numbers(num_cs))
        population.append(individual)
    return population


def mutation_MODE_rand_1_bin(population, index, para):
    l = len(population)
    all_indices = list(range(l))
    all_indices.remove(index)
    random_indices = random.sample(all_indices, 3)
    random.shuffle(random_indices)
    F = para[0]

    # 确保输入是普通列表
    pop_r1 = population[random_indices[1]]
    if hasattr(pop_r1, 'tolist'):
        pop_r1 = pop_r1.tolist()

    pop_r2 = population[random_indices[2]]
    if hasattr(pop_r2, 'tolist'):
        pop_r2 = pop_r2.tolist()

    # 计算差值
    diff = deal_with_list(pop_r1, pop_r2, '-')

    # 计算缩放后的差值
    scaled_diff = []
    for d in diff:
        if hasattr(d, 'tolist'):
            d = d.tolist()
        scaled_diff.append([F * val for val in d])

    pop_r0 = population[random_indices[0]]
    if hasattr(pop_r0, 'tolist'):
        pop_r0 = pop_r0.tolist()

    # 计算最终结果并确保返回普通列表
    result = deal_with_list(pop_r0, scaled_diff, '+')

    # 最终再次检查结果是否为普通列表
    if hasattr(result, 'tolist'):
        result = result.tolist()
    # print("变1出来的result:", result)
    return result


def mutation_MODE_rand_2_bin(population, index, para):
    l = len(population)
    all_indices = list(range(l))
    all_indices.remove(index)
    random_indices = random.sample(all_indices, min(5, len(all_indices)))
    random.shuffle(random_indices)
    F = para[0]

    # 确保所有输入都是普通列表
    for i in range(min(5, len(all_indices))):
        if hasattr(population[random_indices[i]], 'tolist'):
            population[random_indices[i]] = population[random_indices[i]].tolist()

    # 计算: r0 + F * (r1 + r2 - r3 - r4)
    sum1 = deal_with_list(population[random_indices[1]], population[random_indices[2]], '+')
    sum2 = deal_with_list(population[random_indices[3]], population[random_indices[4]], '+')
    diff = deal_with_list(sum1, sum2, '-')

    # 缩放差值
    scaled_diff = []
    for d in diff:
        if hasattr(d, 'tolist'):
            d = d.tolist()
        scaled_diff.append([F * val for val in d])

    # 计算最终结果
    result = deal_with_list(population[random_indices[0]], scaled_diff, '+')

    # 最终检查
    if hasattr(result, 'tolist'):
        result = result.tolist()
    # print("变2出来的result:", result)
    return result

def simple_crossover(population, index, para, mu_ind):
    ind = population[index]
    l = len(ind)
    Cr = para[1]
    new_ind = []
    for i in range(l):
        if random.random() < Cr:
            new_ind.append(mu_ind[i])
        else:
            new_ind.append(ind[i])
    for i in range(len(new_ind)):
        for j in range(len(new_ind[i])):
            if new_ind[i][j] < 0:
                new_ind[i][j] = min(1, -new_ind[i][j])
            elif new_ind[i][j] > 1:
                new_ind[i][j] = max(0, 2 - new_ind[i][j])
        if sum(new_ind[i]) != 1:
            if sum(new_ind[i]) > 0:  # 避免除以零
                new_ind[i] /= sum(new_ind[i])
                new_ind[i] = list(new_ind[i])
            else:  # 如果所有值都为0，则均分
                new_ind[i] = [1.0 / len(new_ind[i]) for _ in range(len(new_ind[i]))]
                new_ind[i] = list(new_ind[i])
    for sub in new_ind:
        sub = list(sub)
    # print("简单交叉出来的result:", new_ind)
    return new_ind


def mutation_cross_current_to_rand_1(population, index, para):
    l = len(population)
    all_indices = list(range(l))
    all_indices.remove(index)
    random_indices = random.sample(all_indices, 3)
    random.shuffle(random_indices)
    F = para[0]
    rand = random.random()

    # 确保所有输入都是普通列表
    current = population[index]
    if hasattr(current, 'tolist'):
        current = current.tolist()

    for i in range(3):
        if hasattr(population[random_indices[i]], 'tolist'):
            population[random_indices[i]] = population[random_indices[i]].tolist()

    # 计算第一部分: current + rand * (r0 - current)
    diff1 = deal_with_list(population[random_indices[0]], current, '-')
    scaled_diff1 = []
    for d in diff1:
        if hasattr(d, 'tolist'):
            d = d.tolist()
        scaled_diff1.append([rand * val for val in d])

    # 计算第二部分: F * (r1 - r2)
    diff2 = deal_with_list(population[random_indices[1]], population[random_indices[2]], '-')
    scaled_diff2 = []
    for d in diff2:
        if hasattr(d, 'tolist'):
            d = d.tolist()
        scaled_diff2.append([F * val for val in d])

    # 组合两部分
    result = deal_with_list(current, scaled_diff1, '+')
    if hasattr(result, 'tolist'):
        result = result.tolist()

    result = deal_with_list(result, scaled_diff2, '+')
    if hasattr(result, 'tolist'):
        result = result.tolist()

    # 确保结果在有效范围内并归一化
    for i in range(len(result)):
        # 处理每个子列表
        if hasattr(result[i], 'tolist'):
            result[i] = result[i].tolist()

        # 约束值范围
        for j in range(len(result[i])):
            if result[i][j] < 0:
                result[i][j] = min(1, -result[i][j])
            elif result[i][j] > 1:
                result[i][j] = max(0, 2 - result[i][j])

        # 归一化
        total = sum(result[i])
        if total > 0:  # 避免除以零
            result[i] = [val / total for val in result[i]]
        else:  # 如果所有值都为0，则均分
            result[i] = [1.0 / len(result[i]) for _ in range(len(result[i]))]
    # print("变异+交叉出来的result:", result)
    return result


def crowding_distance_assignment(population, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice,lmp_dict, cs, cs_bus, od_wait):
    num = len(population)
    distance = [0] * (num + 1)

    sol1 = population.copy()
    sol1 = sorted(sol1, key=lambda x: f1(x, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice))
    distance[0] = distance[num - 1] = math.inf
    for i in range(1, num - 1):
        distance[i] = (distance[i] + (
                    f1(sol1[i + 1], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice) - f1(sol1[i - 1],
                                                                                                                      OD_ratio,
                                                                                                                      anxiety_OD_ratio,
                                                                                                                      od_length,
                                                                                                                      cs_for_choice,
                                                                                                                      anxiety_cs_for_choice))
                       / (f1(sol1[num - 1], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice) - f1(sol1[0],
                                                                                                       OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)))

    sol2 = population.copy()
    sol2 = sorted(sol2, key=lambda x: f2(x, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait))
    distance[0] = distance[num - 1] = math.inf
    for i in range(1, num - 1):
        distance[i] = (distance[i] + (
                    f2(sol2[i + 1], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait) - f2(sol2[i - 1],
                                                                                                                      lmp_dict,
                                                                                                                      OD_ratio,
                                                                                                                      anxiety_OD_ratio,
                                                                                                                      cs,
                                                                                                                      cs_bus,
                                                                                                                      cs_for_choice,
                                                                                                                      anxiety_cs_for_choice,
                                                                                                                      od_length,
                                                                                                                      od_wait))
                       / (f2(sol2[num - 1], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait) - f2(sol2[0],
                                                                                                       lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait)))
    return distance



def dispatch_cs_MODE(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None, P_first = 0):
    od_length, od_wait = process_od_length(real_path_results)
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    population = initialize_MODE(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    params = [[1, 0.1], [1, 0.9], [0.8, 0.2]]

    for loop_cnt in range(max_iter):
        print("第{}代".format(loop_cnt + 1))
        evo_set = population.copy()
        para_indices = list(range(len(params)))
        for i in range(len(population)):
            random.shuffle(para_indices)
            fit = [[0, 0, 0] for _ in range(len(params))]
            new_ind = [[] for _ in range(len(params))]
            mutation_idx0 = para_indices[0]
            mu_ind0 = mutation_MODE_rand_1_bin(population, i, params[mutation_idx0])
            new_ind[0] = simple_crossover(population, i, params[mutation_idx0], mu_ind0)
            fit[0][0] = f1(new_ind[0], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            fit[0][1] = f2(new_ind[0], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait)

            mutation_idx1 = para_indices[1]
            mu_ind1 = mutation_MODE_rand_2_bin(population, i, params[mutation_idx1])
            new_ind[1] = simple_crossover(population, i, params[mutation_idx1], mu_ind1)
            fit[1][0] = f1(new_ind[1], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            fit[1][1] = f2(new_ind[1], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait)

            mutation_idx2 = para_indices[2]
            new_ind[2] = mutation_cross_current_to_rand_1(population, i, params[mutation_idx2])
            fit[2][0] = f1(new_ind[2], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            fit[2][1] = f2(new_ind[2], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait)

            for jj in range(len(params)):
                for kk in range(jj+1, len(params)):
                    if(fit[jj][0] < fit[kk][0] and fit[jj][1] < fit[kk][1]) or (fit[jj][0] <= fit[kk][0] and fit[jj][1] < fit[kk][1]) or (fit[jj][0] < fit[kk][0] and fit[jj][1] <= fit[kk][1]):
                        fit[kk][2] += 1

            for jj in range(len(params)):
                if(fit[jj][2] == 0):
                    evo_set.append(new_ind[jj])
                    break

        front, fit1, fit2 = fast_non_dominated_sorting(evo_set, OD_ratio, anxiety_OD_ratio, cs_for_choice,
                                                       anxiety_cs_for_choice, od_length, od_wait, cs_bus, lmp_dict, cs)
        new_population = []
        if loop_cnt < max_iter - 1:
            for front_cnt in range(len(front)):
                if len(front[front_cnt]) < num_population - len(new_population):
                    for ii in front[front_cnt]:
                        new_population.append(evo_set[ii])
                elif len(front[front_cnt]) == num_population - len(new_population):
                    for ii in front[front_cnt]:
                        new_population.append(evo_set[ii])
                    break
                else:
                    distance = crowding_distance_assignment(evo_set, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice, lmp_dict, cs, cs_bus, od_wait)
                    sorted_indices = sorted(range(len(distance)), key=lambda k: distance[k], reverse=True)
                    for ii in range(num_population - len(new_population)):
                        new_population.append(evo_set[sorted_indices[ii]])
                    break

            population = new_population
            print("population updated")
        else:
            for ii in front[0]:
                new_population.append([f1(evo_set[ii], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice),
                                      f2(evo_set[ii], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait), evo_set[ii]])
            new_population.sort(key=lambda x: x[P_first])
            population = new_population

    print(f"best_solution: {population}")
    return population, cs_for_choice, anxiety_cs_for_choice

##################################################################################################################RCNSGA

def SBX_crossover_R_NSGA2(parent1, parent2, eta=20):
    """
    模拟二进制交叉(SBX)操作符，处理多维列表结构
    """
    rand = random.random()
    if rand == 1:
        return parent1, parent2
    elif rand < 0.5:
        beta = (2 * rand) ** (1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - rand))) ** (1 / (eta + 1))

    # 计算两个子代的参数
    beta_plus = 0.5 * (1 + beta)
    beta_minus = 0.5 * (1 - beta)

    # 计算子代1
    child1 = []
    for i in range(len(parent1)):
        # 计算部分1: beta_plus * parent1[i]
        part1 = [beta_plus * val for val in parent1[i]]
        # 计算部分2: beta_minus * parent2[i]
        part2 = [beta_minus * val for val in parent2[i]]
        # 相加
        child_part = [a + b for a, b in zip(part1, part2)]
        child1.append(child_part)

    # 计算子代2
    child2 = []
    for i in range(len(parent1)):
        # 计算部分1: beta_minus * parent1[i]
        part1 = [beta_minus * val for val in parent1[i]]
        # 计算部分2: beta_plus * parent2[i]
        part2 = [beta_plus * val for val in parent2[i]]
        # 相加
        child_part = [a + b for a, b in zip(part1, part2)]
        child2.append(child_part)

    # 确保子代1的值都在[0,1]范围内并且和为1
    for i in range(len(child1)):
        for j in range(len(child1[i])):
            if child1[i][j] < 0:
                child1[i][j] = 0
            elif child1[i][j] > 1:
                child1[i][j] = 1
        total = sum(child1[i])
        if total > 0:  # 避免除以零
            child1[i] = list([val / total for val in child1[i]])
        else:  # 如果所有值都为0，则均分
            child1[i] = list([1.0 / len(child1[i]) for _ in range(len(child1[i]))])

    # 确保子代2的值都在[0,1]范围内并且和为1
    for i in range(len(child2)):
        for j in range(len(child2[i])):
            if child2[i][j] < 0:
                child2[i][j] = 0
            elif child2[i][j] > 1:
                child2[i][j] = 1
        total = sum(child2[i])
        if total > 0:  # 避免除以零
            child2[i] = list([val / total for val in child2[i]])
        else:  # 如果所有值都为0，则均分
            child2[i] = list([1.0 / len(child2[i]) for _ in range(len(child2[i]))])

    for sub in child1:
        sub = list(sub)
    for sub in child2:
        sub = list(sub)
    return child1, child2


def polynomial_mutation(individual, mutation_prob, eta=20):
    """
    对个体进行多项式变异，处理多维列表结构

    参数:
        individual: 要变异的个体
        mutation_prob: 变异概率
        eta: 分布指数 - 控制变异范围，越大变异范围越小

    返回:
        变异后的个体
    """
    mutated = []
    for sub_list in individual:
        mutated_sub = sub_list.copy()
        for i in range(len(sub_list)):
            # 以mutation_prob的概率对每个元素进行变异
            if random.random() < mutation_prob:
                # 生成随机数用于变异计算
                r = random.random()

                # 计算变异因子 δ
                if r < 0.5:
                    delta = (2 * r) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))

                # 应用变异
                mutated_value = mutated_sub[i] + delta

                # 约束到[0,1]区间
                if mutated_value < 0:
                    mutated_value = 0
                elif mutated_value > 1:
                    mutated_value = 1

                mutated_sub[i] = mutated_value

        # 归一化，确保和为1
        total = sum(mutated_sub)
        if total > 0:  # 避免除以零
            mutated_sub = list([v / total for v in mutated_sub])
        else:  # 如果所有值都为0，则均分
            mutated_sub = list([1.0 / len(mutated_sub) for _ in range(len(mutated_sub))])

        mutated.append(mutated_sub)

    return mutated



def dispatch_cs_R_NSGA2(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None, P_first = 0):
    od_length, od_wait = process_od_length(real_path_results)
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    population = initialize_MODE(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    numbers = list(range(num_population))

    for loop_cnt in range(max_iter):
        print("第{}代".format(loop_cnt + 1))
        random.shuffle(numbers)
        evo_set = population.copy()
        for i in range(0, len(numbers), 2):
            parent1 = population[numbers[i]]
            parent2 = population[numbers[i + 1]]
            child1, child2 = SBX_crossover_R_NSGA2(parent1, parent2)
            evo_set.append(child1)
            evo_set.append(child2)

        fit1_values = []
        fit2_values = []
        for ind in evo_set:
            fit1_values.append(f1(ind, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice))
            fit2_values.append(f2(ind, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                                  anxiety_cs_for_choice, od_length, od_wait))

        front, _, _ = fast_non_dominated_sorting(evo_set, OD_ratio, anxiety_OD_ratio, cs_for_choice,
                                                  anxiety_cs_for_choice, od_length, od_wait, cs_bus, lmp_dict, cs)

        last_front_population = []
        new_population = []
        len_cnt = 0
        if loop_cnt < max_iter - 1:
            for kk in range(0, len(front)):
                if len_cnt == num_population:
                    break
                len_cnt += len(front[kk])
                if len_cnt > num_population:
                    for id in front[kk]:
                        last_front_population.append(evo_set[id])
                    break
                else:
                    for id in front[kk]:
                        new_population.append(evo_set[id])
            if last_front_population:
                ranking = crowding_distance_assignment(evo_set, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice,lmp_dict, cs, cs_bus, od_wait)
                last_front_population = sorted(last_front_population,
                                               key=lambda x: ranking[last_front_population.index(x)])
                new_population += last_front_population[:num_population - len(new_population)]
            population = new_population
        else:
            for ii in front[0]:
                new_population.append([f1(evo_set[ii], OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice),
                                      f2(evo_set[ii], lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice, anxiety_cs_for_choice, od_length, od_wait), evo_set[ii]])
            new_population.sort(key=lambda x: x[P_first])
            population = new_population

    print(f"best_solution: {population}")
    return population, cs_for_choice, anxiety_cs_for_choice

###############################################################################################################MOEAD-AWA

def initialize_MOEAD_spv(num_population):
    eps = 1 / num_population
    vectors = []
    v1 = []
    v2 = []
    for i in range(num_population):
        v1.append(eps * i + 1e-10)
        v2.append(eps * (num_population - i) + 1e-10)

    s1 = sum([1 / v for v in v1])
    s2 = sum([1 / v for v in v2])

    for i in range(num_population):
        vectors.append([1 / v1[i] / s1, 1 / v2[i] / s2])

    return vectors

def find_neighbors(sub_vs, num):# 选取其中最近的num个邻居
    neighbors = []
    for i in range(len(sub_vs)):
        distances = []
        for j in range(len(sub_vs)):
            if i != j:
                dist = math.sqrt((sub_vs[i][1] - sub_vs[j][1]) ** 2 + (sub_vs[i][0] - sub_vs[j][0]) ** 2)
                distances.append((j, dist))
        distances.sort(key=lambda x: x[1])
        neighbors.append([distances[k][0] for k in range(num)])
    return neighbors


def delete_duplicates(population, points, neighbors, spv, nus):
    """
    删除稀疏度最低(拥挤度最高)的nus个解以及对应的权重向量。
    使用前k个最近邻居的距离乘积作为稀疏度度量。

    Args:
        points: 解集列表，每个元素是[individual, fit1_value, fit2_value]
        neighbors: 邻居列表，每个元素包含该解的邻居索引
        spv: 权重向量列表
        nus: 要删除的解的数量

    Returns:
        更新后的points、neighbors和spv
    """
    if nus <= 0 or len(points) <= nus:
        return points, neighbors, spv

    # 计算每个解的稀疏度（使用距离乘积）
    sparsity_degree = [1.0] * len(points)

    for i in range(len(points)):
        # 获取当前点的适应度值
        current_fit1 = points[i][1]
        current_fit2 = points[i][2]

        # 计算与其他解的欧几里得距离
        distances = []
        for j in range(len(points)):
            if i != j:
                other_fit1 = points[j][1]
                other_fit2 = points[j][2]
                # 计算欧几里得距离
                dist = math.sqrt((current_fit1 - other_fit1) ** 2 + (current_fit2 - other_fit2) ** 2)
                distances.append(dist)

        # 排序并取前k个最近邻居的距离
        k = min(2, len(distances))
        distances.sort()

        # 计算距离乘积作为稀疏度度量（距离越小，稀疏度越低）
        for d in distances[:k]:
            if d > 1e-10:  # 避免乘以非常小的值导致下溢
                sparsity_degree[i] *= d
            else:
                # 对于极小的距离，使用一个小的常数以避免乘积为零
                sparsity_degree[i] *= 1e-10

    # 根据稀疏度从小到大排序（稀疏度小的先被删除）
    indices_to_delete = sorted(range(len(sparsity_degree)), key=lambda x: sparsity_degree[x])[:nus]

    # 从大到小删除索引(避免删除后索引变化)
    indices_to_delete.sort(reverse=True)

    # 删除选中的解及对应权重向量
    for idx in indices_to_delete:
        points.pop(idx)
        spv.pop(idx)
        population.pop(idx)

    # 重新计算邻居关系
    new_neighbors = find_neighbors(spv, len(neighbors[0]))

    return points, new_neighbors, spv, population


def generate_new_solutions(REP, nus, population, points, spv, standard_point):
    """
    从存档REP中选择稀疏度最高的nus个解加入当前种群，并生成对应的权重向量

    Args:
        REP: 非支配解集，格式为[(fit1, fit2, individual), ...]
        nus: 要选择的解的数量
        population: 当前种群
        points: 当前种群点集，格式为[[individual, fit1, fit2], ...]
        spv: 当前权重向量集

    Returns:
        更新后的population、points和spv
    """
    if not REP or nus <= 0:
        return population, points, spv

    # 1. 计算REP中每个解的稀疏度
    sparsity_degree = [1.0] * len(REP)

    for i in range(len(REP)):
        # 获取当前解的适应度值
        current_fit1 = REP[i][0]
        current_fit2 = REP[i][1]

        # 计算与REP中其他解的欧几里得距离
        distances = []
        for j in range(len(REP)):
            if i != j:
                other_fit1 = REP[j][0]
                other_fit2 = REP[j][1]
                dist = math.sqrt((current_fit1 - other_fit1) ** 2 + (current_fit2 - other_fit2) ** 2)
                distances.append(dist)

        # 排序并取前k个最近邻居的距离
        k = min(3, len(distances))
        if distances:
            distances.sort()

            # 计算距离乘积作为稀疏度度量（距离越大，稀疏度越高）
            for d in distances[:k]:
                if d > 1e-10:
                    sparsity_degree[i] *= d
                else:
                    sparsity_degree[i] *= 1e-10

    # 2. 根据稀疏度从大到小排序（选择稀疏度高的解）
    sorted_indices = sorted(range(len(sparsity_degree)), key=lambda x: sparsity_degree[x], reverse=True)

    # 3. 选择前nus个最稀疏的解
    selected_indices = sorted_indices[:min(nus, len(REP))]

    # 4. 为选中的解生成权重向量并加入当前种群
    for idx in selected_indices:
        individual = REP[idx][2]
        fit1_value = 1 / (REP[idx][0] - standard_point[0])
        fit2_value = 1 / (REP[idx][1] - standard_point[1])

        # 生成新的权重向量
        # 使用切比雪夫归一化，根据解在目标空间的位置生成权重
        total_fit = fit1_value + fit2_value
        if total_fit > 1e-10:
            new_weight = [fit2_value / total_fit, fit1_value / total_fit]
        else:
            new_weight = [0.5, 0.5]

        # 加入种群
        population.append(individual)
        points.append([individual, fit1_value, fit2_value])
        spv.append(new_weight)

    return population, points, spv


def dispatch_cs_MOEAD(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio=None, P_first = 0):
    def initialize_standard_point(population):
        b1 = float('inf')
        b2 = float('inf')
        point_set = []
        for ind in population:
            x = f1(ind, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            y = f2(ind, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                            anxiety_cs_for_choice, od_length, od_wait)
            b1 = min(b1, x)
            b2 = min(b2, y)
            point_set.append([ind, x, y])

        return [b1 - 1e-7, b2 - 1e-7], point_set

    def chebyshev(individual, sub_v, st_p):
        ch1 = sub_v[0] * abs(f1(individual, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice) - st_p[0])
        ch2 = sub_v[1] * abs(f2(individual, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                            anxiety_cs_for_choice, od_length, od_wait) - st_p[1])

        return max(ch1, ch2)

    num_neighbor = int(0.1 * num_population)
    prob_neighbor = 0.9
    od_length, od_wait = process_od_length(real_path_results)
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    population = initialize_MODE(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    spv = initialize_MOEAD_spv(num_population)
    standard_point, points = initialize_standard_point(population)
    prob_mutation = 1 / len(population) / len(population[0])
    wag = 50
    nus = int(0.05 * num_population)
    G_max = int(0.05 * num_population * 0.8)
    REP = []
    REP_max = 1.5 * num_population

    for loop_cnt in range(max_iter):
        print("第{}代".format(loop_cnt + 1))
        neighbors = find_neighbors(spv, num_neighbor)
        for i in range(num_population):
            print("处理第{}个".format(i + 1))
            r = random.random()
            r1 = num_population
            r2 = num_population
            c = num_population
            P = []
            if r < prob_neighbor:
                P = neighbors[i]
                r1, r2 = random.sample(neighbors[i], 2)
                c = len(P)
            else:
                r1, r2 = random.sample(range(num_population), 2)
                P = list(range(num_population))
            parent1 = population[r1]
            parent2 = population[r2]
            child1, child2 = SBX_crossover_R_NSGA2(parent1, parent2)
            rr = random.random()
            new_ind = []
            if rr < 0.5:
                if rr < prob_mutation:
                    new_ind = polynomial_mutation(child1, prob_mutation)
                else:
                    new_ind = child1
            else:
                if rr >= 1 - prob_mutation:
                    new_ind = polynomial_mutation(child2, prob_mutation)
                else:
                    new_ind = child2

            n1 = f1(new_ind, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            n2 = f2(new_ind, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                                  anxiety_cs_for_choice, od_length, od_wait)
            if n1 < standard_point[0]:
                standard_point[0] = n1
            if n2 < standard_point[1]:
                standard_point[1] = n2

            while c > 0 and len(P) > 0:
                # print(f"P:{P}")
                j = P[random.choice(range(len(P)))]
                P.remove(j)
                if j == i:
                    continue
                if chebyshev(new_ind, spv[j], points[j][1:]) < chebyshev(population[j], spv[j], points[j][1:]):
                    population[j] = new_ind
                    points[j][0] = new_ind
                    points[j][1] = n1
                    points[j][2] = n2
                    c -= 1

        if loop_cnt % wag == 0 and loop_cnt >= G_max:
            print(f"第{loop_cnt}代更新REP")
            if loop_cnt == 0:
                for point in points:
                    REP.append([point[1], point[2], point[0]])
            else:
                REP = update_REP(REP, population, OD_ratio, anxiety_OD_ratio, cs_for_choice, anxiety_cs_for_choice,
                                 od_length, od_wait, cs_bus, lmp_dict, cs, REP_max)

            REP = cleanup_REP(REP)
            points, new_neighbors, spv, population = delete_duplicates(population, points, neighbors, spv, nus)
            population, points, spv = generate_new_solutions(REP, nus, population, points, spv, standard_point)

    print(f"best_solution: {REP}")
    return REP, cs_for_choice, anxiety_cs_for_choice


################################################################################################################MODED
def dispatch_cs_MODED(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict,
                      max_iter, OD_ratio, anxiety_OD_ratio=None, P_first=0):
    od_length, od_wait = process_od_length(real_path_results)
    cs_for_choice = process_cs(OD_ratio, cs, num_cs, od_length)
    anxiety_cs_for_choice = process_cs(anxiety_OD_ratio, cs, num_cs, od_length, 0)
    population = initialize_MODE(num_population, num_cs, len(OD_ratio.keys()) + len(anxiety_OD_ratio.keys()))
    nus = int(0.05 * num_population)

    # 初始化权重向量和邻居
    spv = initialize_MOEAD_spv(num_population)
    neighbors = find_neighbors(spv, int(num_population * 0.1))

    # 初始化标准点和点集
    standard_point = [float('inf'), float('inf')]
    point_set = []
    for ind in population:
        x = f1(ind, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
        y = f2(ind, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
               anxiety_cs_for_choice, od_length, od_wait)
        standard_point[0] = min(standard_point[0], x) - 1e-10
        standard_point[1] = min(standard_point[1], y) - 1e-10
        point_set.append([ind, x, y])

    # DE算法参数
    F = 0.5  # 缩放因子
    CR = 0.9  # 交叉概率

    # 非支配解集
    REP = []

    for loop_cnt in range(max_iter):
        print("第{}代".format(loop_cnt + 1))

        # 对每个权重向量进行迭代
        for i in range(len(population)):
            # 从邻居中随机选择三个不同的个体
            neighbor_indices = neighbors[i].copy()
            random.shuffle(neighbor_indices)
            idx1, idx2, idx3 = neighbor_indices[:3]

            # 差分进化: 变异操作 (DE/rand/1)
            mutation_vector = []
            for j in range(len(population[idx1])):
                # 计算: x_r1 + F * (x_r2 - x_r3)
                diff = deal_with_list(population[idx2][j], population[idx3][j], '-')
                scaled_diff = [F * val for val in diff]
                result = deal_with_list(population[idx1][j], scaled_diff, '+')

                # 确保归一化
                for k in range(len(result)):
                    # 处理超出范围的值
                    if result[k] < 0:
                        result[k] = min(1, -result[k])
                    elif result[k] > 1:
                        result[k] = max(0, 2 - result[k])

                # 归一化
                total = sum(result)
                if total > 0:
                    result = [val / total for val in result]
                else:
                    result = [1.0 / len(result) for _ in range(len(result))]

                mutation_vector.append(result)

            # 差分进化: 交叉操作 (二项式交叉)
            trial_vector = []
            for j in range(len(population[i])):
                child = []
                for k in range(len(population[i][j])):
                    if random.random() < CR:
                        child.append(mutation_vector[j][k])
                    else:
                        child.append(population[i][j][k])

                # 确保归一化
                total = sum(child)
                if total > 0:
                    child = [val / total for val in child]
                else:
                    child = [1.0 / len(child) for _ in range(len(child))]

                trial_vector.append(child)

            # 评估新个体
            f1_trial = f1(trial_vector, OD_ratio, anxiety_OD_ratio, od_length, cs_for_choice, anxiety_cs_for_choice)
            f2_trial = f2(trial_vector, lmp_dict, OD_ratio, anxiety_OD_ratio, cs, cs_bus, cs_for_choice,
                          anxiety_cs_for_choice, od_length, od_wait)

            # 使用切比雪夫方法评估
            g_old = max(spv[i][0] * (point_set[i][1] - standard_point[0]),
                        spv[i][1] * (point_set[i][2] - standard_point[1]))

            g_new = max(spv[i][0] * (f1_trial - standard_point[0]),
                        spv[i][1] * (f2_trial - standard_point[1]))

            # 选择
            if g_new <= g_old:
                population[i] = trial_vector
                point_set[i] = [trial_vector, f1_trial, f2_trial]
                standard_point[0] = min(standard_point[0], f1_trial) - 1e-10
                standard_point[1] = min(standard_point[1], f2_trial) - 1e-10

                # 更新邻居的解
                for j in neighbors[i]:
                    if j != i:
                        g_old_j = max(spv[j][0] * (point_set[j][1] - standard_point[0]),
                                      spv[j][1] * (point_set[j][2] - standard_point[1]))
                        g_new_j = max(spv[j][0] * (f1_trial - standard_point[0]),
                                      spv[j][1] * (f2_trial - standard_point[1]))

                        if g_new_j <= g_old_j:
                            population[j] = trial_vector
                            point_set[j] = [trial_vector, f1_trial, f2_trial]

            # 更新REP
            is_dominated = False
            dominates_someone = False
            to_remove = []

            for k in range(len(REP)):
                # 检查新解是否被REP中的解支配
                if (REP[k][0] < f1_trial and REP[k][1] < f2_trial) or \
                        (REP[k][0] <= f1_trial and REP[k][1] < f2_trial) or \
                        (REP[k][0] < f1_trial and REP[k][1] <= f2_trial):
                    is_dominated = True
                    break

                # 检查新解是否支配REP中的解
                if (f1_trial < REP[k][0] and f2_trial < REP[k][1]) or \
                        (f1_trial <= REP[k][0] and f2_trial < REP[k][1]) or \
                        (f1_trial < REP[k][0] and f2_trial <= REP[k][1]):
                    to_remove.append(k)
                    dominates_someone = True

            # 更新REP
            if not is_dominated:
                if dominates_someone:
                    # 从后向前移除被支配的解
                    for k in sorted(to_remove, reverse=True):
                        REP.pop(k)
                REP.append((f1_trial, f2_trial, trial_vector))

        # 每代结束后的操作
        if loop_cnt % 10 == 0 and loop_cnt > 0:
            # 删除一些解并从REP中添加新解
            point_set, new_neighbors, spv, population = delete_duplicates(population, point_set, neighbors, spv, nus)
            population, point_set, spv = generate_new_solutions(REP, nus, population, point_set, spv, standard_point)
            neighbors = find_neighbors(spv, int(num_population * 0.1))

        RRRRRREP = REP.copy()
        RRRRRREP.sort(key=lambda x: x[P_first])
        print(RRRRRREP[0])
        print(RRRRRREP[-1])
        print(444444444444444444444444444444444444444)

    # 最终处理REP
    REP = cleanup_REP(REP)
    REP.sort(key=lambda x: x[P_first])

    print(f"best_solution: {REP[0]}")
    return REP, cs_for_choice, anxiety_cs_for_choice

##################################################################################################################MOEAD
