# Jyh-Da Wei and D. T. Lee, "A new approach to the traveling salesman problem using genetic algorithms with priority encoding," Proceedings of the 2004 Congress on Evolutionary Computation (IEEE Cat. No.04TH8753), Portland, OR, USA, 2004, pp. 1457-1464 Vol.2, doi: 10.1109/CEC.2004.1331068.
#实验1用data1，选择算子用轮盘赌，种群大小100，变异概率0.2，采用均匀交叉算子
#实验2用rd100和ch150
#原文实现是1高优先级，2低优先级

import math
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor


def euclidean_distance(x1, y1, x2, y2):
    # 欧式距离
    return math.sqrt(((x1 - x2) ** 2  + (y1 - y2) ** 2)/ 1.0)


data1 = {
    1: (0.047, 0.692), 2: (0.054, 0.577), 3: (0.095, 0.004), 4: (0.146, 0.855), 5: (0.157, 0.319), 6: (0.259, 0.085),
    7: (0.326, 0.748), 8: (0.336, 0.53), 9: (0.399, 0.143), 10: (0.399, 0.275), 11: (0.436, 0.367), 12: (0.525, 0.222),
    13: (0.534, 0.031), 14: (0.542, 0.602), 15: (0.586, 0.982), 16: (0.684, 0.292), 17: (0.694, 0.585), 18: (0.732, 0.811),
    19: (0.891, 0.082), 20: (0.946, 0.728)
}

def generate_distance_patrix(data):
    num_nodes = len(data)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i != j:
                distance_matrix[i - 1, j - 1] = euclidean_distance(data[i][0], data[i][1], data[j][0], data[j][1])
            else:
                distance_matrix[i - 1, j - 1] = float('inf')
    return distance_matrix

def generate_individuals(num, l, w):
    individuals = []
    for i in range(num):
        individual = np.zeros((l, w), dtype=int)
        for j in range(l):
            for k in range(w):
                if j != k:
                    individual[j, k] = random.randint(1, 2)
                else:
                    individual[j, k] = -1
        individuals.append(individual)
    return individuals


def DENN(distance_matrix, priority_matrix):
    #排序
    rows, cols = distance_matrix.shape
    indexed_values = []
    for i in range(rows):
        for j in range(cols):
            if i != j:
                indexed_values.append((priority_matrix[i, j], distance_matrix[i, j], [i, j]))
    indexed_values.sort(key=lambda x: (x[0], x[1]))
    sorted_indices = [index for [priority, distance, index] in indexed_values]
    # print(sorted_indices)
    # print(1111111111111111111111111111111111111111111111111111111)

    #迭代路径
    path_nodes = sorted_indices[0].copy()
    end_nodes = sorted_indices[0].copy()
    # print(path_nodes)
    while(len(path_nodes) < rows):
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i][0] in end_nodes and sorted_indices[i][1] not in path_nodes:
                if sorted_indices[i][0] == end_nodes[1]:
                    end_nodes[1] = sorted_indices[i][1]
                    path_nodes.append(sorted_indices[i][1])
                    # print(path_nodes, end_nodes)
                    break

                else:
                    end_nodes[0] = sorted_indices[i][1]
                    path_nodes.insert(0, sorted_indices[i][1])
                    # print(path_nodes, end_nodes)
                    break

            elif sorted_indices[i][1] in end_nodes and sorted_indices[i][0] not in path_nodes:
                if sorted_indices[i][1] == end_nodes[1]:
                    end_nodes[1] = sorted_indices[i][0]
                    path_nodes.append(sorted_indices[i][0])
                    # print(path_nodes, end_nodes)
                    break
                else:
                    end_nodes[0] = sorted_indices[i][0]
                    path_nodes.insert(0, sorted_indices[i][0])
                    # print(path_nodes, end_nodes)
                    break
    path_nodes.append(path_nodes[0])
    return path_nodes


# def SEF(distance_matrix):
#     # 排序
#     rows, cols = distance_matrix.shape
#     indexed_values = []
#     for i in range(rows):
#         for j in range(cols):
#             if i != j:
#                 indexed_values.append([distance_matrix[i, j], [i, j]])
#     indexed_values.sort(key=lambda x: x[0])
#     sorted_indices = [index for [value, index] in indexed_values]


def selection(population, fitness):
    # 轮盘赌选择
    total_fitness = sum(fitness)
    probability = [f / total_fitness for f in fitness]
    probability_sum = np.cumsum(probability)
    # print(fitness)
    # print(probability)
    # print(probability_sum)
    new_population = []
    for i in range(len(population)):
        r = random.random()
        for j in range(len(probability_sum)):
            if r < probability_sum[j]:
                new_population.append(population[j])
                break
    return new_population


def crossover(parent1, parent2):
    # 均匀交叉算子
    child1 = np.zeros(parent1.shape)
    child2 = np.zeros(parent1.shape)
    rows, cols = parent1.shape
    for i in range(rows):
        for j in range(cols):
            if i != j:
                if random.random() < 0.5:
                    child1[i][j] = parent1[i][j]
                    child2[i][j] = parent2[i][j]
                else:
                    child1[i][j] = parent2[i][j]
                    child2[i][j] = parent1[i][j]

            else:
                child1[i][j] = float('inf')
                child2[i][j] = float('inf')
    # for i in range(rows):
    #     for j in range(cols):
    #         if i != j:
    #             if random.random() < 0.5:
    #                 child2[i][j] = parent1[i][j]
    #             else:
    #                 child2[i][j] = parent2[i][j]
    #         else:
    #             child2[i][j] = float('inf')


    return child1, child2

def mutation(individual, mutation_rate):
    # 交换变异算子
    rows, cols = individual.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < mutation_rate:
                individual[i][j] = individual[i][j] + 1
                if individual[i][j] > 2:
                    individual[i][j] = 1
    return individual


iter = 100
population_size = 100
mutation_rate = 0.2

distance_matrix = generate_distance_patrix(data1)
population = generate_individuals(population_size, len(data1), len(data1))
print(distance_matrix)
# print(111)
# print(population)
best_fit = float('inf')
for it in range(iter):
    print(f"iter: {it}")
    fitness = []
    for individual in population:
        path = DENN(distance_matrix, individual)
        # print(path)
        # print(2222222222222222222222222222222222)
        fitness.append(sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)))

    if min(fitness) < best_fit:
        best_individual = population[fitness.index(min(fitness))]
        best_fit = min(fitness)

    selected_population = selection(population, fitness)
    new_population = []
    print(best_fit)
    # print(len(population))
    # print(len(selected_population))
    for i in range(int(population_size / 2) - 1):
        child1, child2 = crossover(selected_population[i * 2], selected_population[i * 2 + 1])
        new_population.append(mutation(child1, mutation_rate))
        new_population.append(mutation(child2, mutation_rate))
    population = new_population

print(best_fit)