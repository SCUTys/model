import numpy as np
import simuplus
import math
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import pandapower as pp
# from TNplus import DispatchCenter

#NSGA2编码：对每辆车预处理车辆没电前可到达并且O-站和站-D的距离之和最短的2^k个站点，排序后按照二进制编码其下标，每辆车的编码长度为k，总长就是车数*k
#
#
#

def process_od_length(real_path_results):
    od_length = {}
    for od, result in real_path_results.items():
        if od[0] == od[1]: continue
        sum = div = 0
        for i in range(len(result) - 1):
            sum += (result[i][1] + result[i][2]) / (i + 1)
            div += 1 / (i + 1)
        sum /= div
        od_length[od] = sum
        od_length[(od[0], od[0])] = 0
        od_length[(od[1], od[1])] = 0

    od_wait = {}
    for od, result in real_path_results.items():
        sum = div = 0
        for i in range(len(result) - 1):
            sum += result[i][2] / (i + 1)
            div += 1 / (i + 1)
        sum /= div
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

    best_solution = population[0]
    bit_per_vehicle = math.ceil(math.log2(num_cs))
    real_cs_ids = []
    for i in range(len(charge_v)):
        cs_index = 0
        for j in range(bit_per_vehicle):
            cs_index += best_solution[i * bit_per_vehicle + j] * 2 ** j
        real_cs_ids.append(cs_for_choice[i][cs_index])

    return population[0], cs_for_choice, real_cs_ids














# def dispatch_cs_nsga2(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter):
#
#     def process_od_length():
#         od_length = {}
#         for od, result in real_path_results.items():
#             if od[0] == od[1]: continue
#             sum = div = 0
#             # print("result", result)
#             for i in range(len(result) - 1):
#                 sum += (result[i][1] +  result[i][2]) / (i + 1)
#                 div += 1 / (i + 1)
#             sum /= div
#             od_length[od] = sum
#             od_length[(od[0], od[0])] = 0
#             od_length[(od[1], od[1])] = 0
#
#         od_wait = {}
#         for od, result in real_path_results.items():
#             sum = div = 0
#             for i in range(len(result) - 1):
#                 sum += result[i][2] / (i + 1)
#                 div += 1 / (i + 1)
#             sum /= div
#             od_wait[od] = sum
#         return od_length, od_wait
#
#     def process_cs(od_length):
#         cs_for_choice = []
#         for od in charge_od:
#             o, d = od
#             cs.sort(key=lambda x: od_length[(o, x)] + od_length[(x, d)])
#             cs_for_choice.append(cs[:num_cs])
#         return cs_for_choice
#
#     def initialize(num_v):
#         bit_per_vehicle = math.ceil(math.log2(num_cs))
#         population = []
#         for i in range(num_population):
#             sol = []
#             for j in range(num_v * bit_per_vehicle):
#                 cs = random.randint(0, 1)
#                 sol.append(cs)
#             population.append(sol)
#         return population
#
#
#
#     def f1(sol, bit_per_vehicle, cs_for_choice, od_length, beta=1):
#         cost = 0
#         for i in range(int(len(sol) / bit_per_vehicle)):
#             # print("在f1")
#             # print(len(sol), bit_per_vehicle)
#             cs_index = 0
#             for j in range(bit_per_vehicle):
#                 cs_index += sol[i * bit_per_vehicle + j] * 2 ** j
#             cs_id = cs_for_choice[i][cs_index]
#             o, d = charge_od[i]
#             cost += od_length[(o, cs_id)] + od_length[(cs_id, d)] * beta
#         return cost
#
#     def f2(sol, bit_per_vehicle, cs_for_choice, od_length, od_wait):
#         lmp = {}
#         bus_node = {}
#         for id in cs_bus:
#             lmp[id] = lmp_dict[id]
#         for i in range(len(cs)):
#             bus_node[cs[i]] = cs_bus[i]
#         sum_value= sum(lmp.values())
#         for value in lmp.values():
#             value /= sum_value
#
#         cost = 0
#         # print("lmp", lmp)
#         for i in range(int(len(sol) / bit_per_vehicle)):
#             v_id = charge_v[i]
#             vehicle = center.vehicles[v_id]
#             cs_index = 0
#             for j in range(bit_per_vehicle):
#                 cs_index += sol[i * bit_per_vehicle + j] * 2 ** j
#             cs_id = cs_for_choice[i][cs_index]
#             o, d = charge_od[i]
#             if cs_id == o:
#                 cost += lmp[bus_node[cs_id]] * (vehicle.Emax - vehicle.E)
#             else:
#                 cost += lmp[bus_node[cs_id]] * (vehicle.Emax - vehicle.E - od_length[(o, cs_id)] * vehicle.Edrive - od_wait[(o, cs_id)] * vehicle.Ewait)
#
#         return cost
#
#     def single_point_crossover(parent1, parent2, bit_per_gene):
#         num_gene = len(parent1) / bit_per_gene
#         point = random.randint(1, num_gene - 1) * bit_per_gene
#         child1 = parent1[:point] + parent2[point:]
#         child2 = parent2[:point] + parent1[point:]
#         return child1, child2
#
#     def bitwise_mutation(sol):
#         for i in range(len(sol)):
#             if random.random() < 1 / len(sol):
#                 sol[i] = 1 - sol[i]
#         return sol
#
#     def compare_individuals(args):
#         i, j, population, bit_per_vehicle, cs_for_choice, od_length, od_wait = args
#         sol = population[i]
#         sol_f1 = f1(sol, bit_per_vehicle, cs_for_choice, od_length)
#         sol_f2 = f2(sol, bit_per_vehicle, cs_for_choice, od_length, od_wait)
#         cmp_f1 = f1(population[j], bit_per_vehicle, cs_for_choice, od_length)
#         cmp_f2 = f2(population[j], bit_per_vehicle, cs_for_choice, od_length, od_wait)
#
#         if (sol_f1 < cmp_f1 and sol_f2 < cmp_f2) or (sol_f1 <= cmp_f1 and sol_f2 < cmp_f2) or (
#                 sol_f1 < cmp_f1 and sol_f2 <= cmp_f2):
#
#             return (i, j)
#         elif (sol_f1 > cmp_f1 and sol_f2 > cmp_f2) or (sol_f1 >= cmp_f1 and sol_f2 > cmp_f2) or (
#                 sol_f1 > cmp_f1 and sol_f2 >= cmp_f2):
#             return (j, i)
#         return None
#
#     def fast_non_dominated_sorting(population, bit_per_vehicle, cs_for_choice, od_length, od_wait, max_workers=4):
#         num = len(population)
#         S = [[] for _ in range(num)]
#         n = [0] * num
#         rank = [0] * num
#         front = [[]]
#
#         def compare_individuals(i, j):
#             sol = population[i]
#             sol_f1 = f1(sol, bit_per_vehicle, cs_for_choice, od_length)
#             sol_f2 = f2(sol, bit_per_vehicle, cs_for_choice, od_length, od_wait)
#             cmp_f1 = f1(population[j], bit_per_vehicle, cs_for_choice, od_length)
#             cmp_f2 = f2(population[j], bit_per_vehicle, cs_for_choice, od_length, od_wait)
#
#             if (sol_f1 < cmp_f1 and sol_f2 < cmp_f2) or (sol_f1 <= cmp_f1 and sol_f2 < cmp_f2) or (
#                     sol_f1 < cmp_f1 and sol_f2 <= cmp_f2):
#                 print(f"{i} dominates {j}")
#                 return (i, j)
#             elif (sol_f1 > cmp_f1 and sol_f2 > cmp_f2) or (sol_f1 >= cmp_f1 and sol_f2 > cmp_f2) or (
#                     sol_f1 > cmp_f1 and sol_f2 >= cmp_f2):
#                 print(f"{j} dominates {i}")
#                 return (j, i)
#             print(f"None dominates {i} and {j}")
#             return None
#
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = []
#             for i in range(num - 1):
#                 for j in range(i + 1, num):
#                     futures.append(executor.submit(compare_individuals, i, j))
#
#             for future in futures:
#                 result = future.result()
#                 if result:
#                     i, j = result
#                     S[i].append(j)
#                     n[j] += 1
#
#         for i in range(num):
#             if n[i] == 0:
#                 front[0].append(i)
#                 rank[i] = 1
#
#         f = 0
#         while front[f]:
#             Q = []
#             for i in front[f]:
#                 for j in S[i]:
#                     n[j] -= 1
#                     if n[j] == 0:
#                         Q.append(j)
#                         rank[j] = f + 1
#             f += 1
#             front.append(Q)
#         print("front", front)
#         return front
#
#
#     def crowding_distance_assignment(population, bit_per_vehicle, cs_for_choice, od_length, od_wait):
#         num = len(population)
#         distance = [0] * (num + 1)
#
#         sol1 = population.copy()
#         sol1 = sorted(sol1, key=lambda x: f1(x,bit_per_vehicle, cs_for_choice, od_length))
#         distance[0] = distance[num - 1] = math.inf
#         for i in range(1, num - 1):
#             distance[i] = (distance[i] + (f1(sol1[i + 1], bit_per_vehicle, cs_for_choice, od_length) - f1(sol1[i - 1], bit_per_vehicle, cs_for_choice, od_length))
#                                             / (f1(sol1[num - 1], bit_per_vehicle, cs_for_choice, od_length) - f1(sol1[0], bit_per_vehicle, cs_for_choice, od_length)))
#
#         sol2 = population.copy()
#         sol2 = sorted(sol2, key=lambda x: f2(x, bit_per_vehicle, cs_for_choice, od_length, od_wait))
#         distance[0] = distance[num - 1] = math.inf
#         for i in range(1, num - 1):
#             distance[i] = (distance[i] + (f2(sol2[i + 1], bit_per_vehicle, cs_for_choice, od_length, od_wait) - f2(sol2[i - 1], bit_per_vehicle, cs_for_choice, od_length, od_wait))
#                            / (f2(sol2[num - 1], bit_per_vehicle, cs_for_choice, od_length, od_wait) - f2(sol2[0], bit_per_vehicle, cs_for_choice, od_length, od_wait)))
#
#         return distance
#
#     od_length, od_wait = process_od_length()
#     cs_for_choice = process_cs(od_length)
#     population = initialize(len(charge_v))
#     print("nsga开始进化")
#     for i in range(max_iter):
#         print("第", i, "代")
#         for j in range(len(population)):
#             if j % 2 == 0:
#                 parent1 = population[j]
#                 parent2 = population[j + 1]
#                 child1, child2 = single_point_crossover(parent1, parent2, math.ceil(math.log2(num_cs)))
#                 population.append(child1)
#                 population.append(child2)
#         for j in range(len(population)):
#             if random.random() < 1 / len(population):
#                 population[j] = bitwise_mutation(population[j])
#         front = fast_non_dominated_sorting(population, math.ceil(math.log2(num_cs)), cs_for_choice, od_length, od_wait)
#         last_front_population = []
#         new_population = []
#         len_cnt = 0
#         for kk in range(0, len(front)):
#             if len_cnt == num_population:
#                 break
#             len_cnt += len(front[kk])
#             if len_cnt > num_population:
#                 for id in front[kk]:
#                     last_front_population.append(population[id])
#                 break
#             else:
#                 for id in front[kk]:
#                     new_population.append(population[id])
#         if last_front_population:
#             ranking = crowding_distance_assignment(last_front_population, math.ceil(math.log2(num_cs)), cs_for_choice, od_length, od_wait)
#             last_front_population = sorted(last_front_population, key=lambda x: ranking[last_front_population.index(x)])
#             new_population += last_front_population[:num_population - len(new_population)]
#         population = new_population
#
#     return population[0], cs_for_choice


























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