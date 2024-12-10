import numpy as np
import simuplus
import math
from TNplus import ChargeStation, Vehicle, DispatchCenter, Edge, Node
cs_SF = [1, 5, 11, 13, 15, 20]
cs_EMA = [6, 10, 11, 17, 19, 22, 23, 25, 27, 29, 30, 33, 34, 38, 40, 42, 44, 47, 48, 49, 52, 57, 60, 63, 65, 69]
cs = cs_SF
T = 10

class SPEA:
    def __init__(self, pop_size, n_gen, n_path1, n_path2, n_cs, n_power, crossover_prob=0.9, mutation_prob=0.1):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_path1 = n_path1
        self.n_path2 = n_path2
        self.n_cs = n_cs
        self.n_power = n_power
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self, n_path1, n_path2, n_cs, n_power):
        population = np.zeros((self.pop_size, 4), dtype=int)
        population[:, 0] = np.random.randint(0, n_path1 + 1, self.pop_size)
        population[:, 1] = np.random.randint(0, n_path2 + 1, self.pop_size)
        population[:, 2] = np.random.randint(1, n_cs + 1, self.pop_size)
        population[:, 3] = np.random.randint(1, n_power + 1, self.pop_size)
        return population

    def evaluate(self, population):
        # Placeholder for actual evaluation function
        return np.random.rand(len(population), 2)

    def fitness_assignment(self, population, objectives):
        # Placeholder for actual fitness assignment
        return np.sum(objectives, axis=1)

    def selection(self, population, fitness):
        selected_indices = np.argsort(fitness)[:self.pop_size]
        return population[selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        return individual

    def run(self, n_var):
        population = self.initialize_population(n_var)
        for gen in range(self.n_gen):
            objectives = self.evaluate(population)
            fitness = self.fitness_assignment(population, objectives)
            selected_population = self.selection(population, fitness)
            new_population = []
            while len(new_population) < self.pop_size:
                parents = selected_population[np.random.choice(len(selected_population), 2, replace=False)]
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population.append(self.mutation(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutation(child2))
            population = np.array(new_population)
        return population

# # Example usage
# spea = SPEA(pop_size=100, n_gen=200)
# final_population = spea.run(n_var=10)
# print(final_population)


class NSGA2:
    def __init__(self, v_charge, center, batch_size, path_result, pop_size, n_gen, n_path1, n_path2, n_cs, n_power, eps = 0, crossover_prob=0.9, mutation_prob=0.1, cal_t=10):
        self.cal_t = cal_t
        self.v_charge = v_charge
        self.center = center
        self.path_result = path_result
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_path1 = n_path1
        self.n_path2 = n_path2
        self.n_cs = n_cs
        self.n_power = n_power
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eps = eps

    def initialize_population(self):
        population = np.zeros((self.pop_size, 4 * self.batch_size), dtype=int)
        for i in range(self.batch_size):
            population[:, 4 * i] = np.random.randint(0, self.n_path1 + 1, self.pop_size)
            population[:, 4 * i + 1] = np.random.randint(0, self.n_path2 + 1, self.pop_size)
            population[:, 4 * i + 2] = np.random.randint(1, self.n_cs + 1, self.pop_size)
            population[:, 4 * i + 3] = np.random.randint(1, self.n_power + 1, self.pop_size)
        return population

    def evaluate(self, population):
        road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
                       range(1, simuplus.num_nodes + 1)}
        cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
        cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
        node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
        eval = []
        for individual in population:
            valid = 1
            for i in range(self.batch_size):
                path1 = individual[4 * i]
                path2 = individual[4 * i + 1]
                c = individual[4 * i + 2]
                power = individual[4 * i + 3]
                vehicle_id = self.v_charge[i]
                vehicle = self.center.vehicles[vehicle_id]
                O = vehicle.origin
                D = vehicle.destination

                if path1 == 0:
                    if O not in cs:
                        eval.append(-1)
                        valid = 0
                        break
                    else:
                        path1_result = []
                else:
                    path1_result = self.path_result[(O, c)][path1]

                if path2 == 0:
                    if D not in cs:
                        eval.append(-1)
                        valid = 0
                        break
                    else:
                        path2_result = []
                else:
                    path2_result = self.path_result[(c, D)][path2]

                path = path1_result + path2_result

                c_result = cs[c - 1]
                power_result = list(self.center.ChargeStation[c].pile.keys())[power - 1]
                ssum = 0
                path_id = self.center.calculate_path(path)

                charge_s = self.center.ChargeStation[c_result]
                if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
                    avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
                else:
                    avg_charge = 1
                charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
                                                       charge_s.capacity * charge_s.pile[power_result] / sum(
                                                           charge_s.pile.values()),
                                                       charge_s.v_arrive[power_result] / T, avg_charge)
                Ecost = charge_t1 * vehicle.Ewait
                for idindex in range(path_id):
                    Ecost += self.center.edges[idindex].calculate_time() * vehicle.Edrive
                    if Ecost > vehicle.E:
                        valid = 0
                        break
                    elif idindex < len(path_id) - 1:
                        Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex],
                                                                                     path_id[idindex + 1])
                        if Ecost > vehicle.E:
                            valid = 0
                            break

                if valid == 0:
                    eval.append(-1)
                    break

                for index in range(0, len(path) - 1):
                    if ssum >= self.cal_t:
                        break
                    if path[index] != c:
                        node = self.center.nodes[path[index + 1]]
                        edge = self.center.edges[path_id[index]]
                        for j in range(math.ceil(ssum), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                                            self.cal_t)):
                            road_counts[(path[index], path[index + 1])][j] += 1
                        for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                                                                    self.cal_t)):
                            node_counts[path[index + 1]][q] += 1
                        ssum += edge.calculate_time()
                    else:
                        charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
                        for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
                            cs_qcounts[c][k] += 1
                        for k in range(math.ceil(ssum + charge_t1),
                                       min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
                            cs_pcounts[c][k] += 1
                        ssum += charge_t1 + charge_t2

            if valid == 0:
                continue

            for m in range(self.cal_t):
                for c_s in cs:
                    c_station = self.center.ChargeStation[c_s]
                    charge_sum = sum(len(v) for v in c_station.charge.values())
                    queue_sum = sum(len(v) for v in c_station.queue.values())
                    if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.ChargeStation[
                        c_s].capacity:
                        valid = 0
                        break
            if valid == 0:
                eval.append(-1)
                continue

            eval_s = 0
            for m in range(self.cal_t):
                for edge in self.center.edges.values():
                    eval_s += (edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][
                        m]) * edge.calculate_time()
                for node in self.center.nodes.values():
                    for p, is_wait in node.wait:
                        eval_s += is_wait + node_counts[node.id][m]
                for charge_station in self.center.charge_stations.values():
                    if list(charge_station.charge.values()) != [[]]:
                        for ipair in charge_station.charge.values():
                            for i in ipair:
                                eval_s += i[1] + cs_pcounts[charge_station.id][m]
                    for p, n in charge_station.pile.items():
                        length = len(charge_station.queue[p])
                        if length > 0:
                            for i, time in charge_station.queue[p]:
                                eval_s += (length + cs_qcounts[charge_station.id][m]) * time

            eval.append(eval_s)
        return eval

    def non_dominated_sorting(self, objectives):
        num_individuals = len(objectives)
        domination_count = np.zeros(num_individuals)
        dominated_solutions = [[] for _ in range(num_individuals)]
        rank = np.zeros(num_individuals)

        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j:
                    if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                        dominated_solutions[i].append(j)
                    elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = 0

        fronts = [[]]
        for i in range(num_individuals):
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = current_front + 1
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, objectives, front):
        distance = np.zeros(len(front))
        for m in range(objectives.shape[1]):
            sorted_indices = np.argsort(objectives[front, m])
            distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf
            for k in range(1, len(front) - 1):
                distance[sorted_indices[k]] += (objectives[front[sorted_indices[k + 1]], m] - objectives[front[sorted_indices[k - 1]], m])
        return distance

    def selection(self, population, fronts, objectives):
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > self.pop_size:
                distances = self.crowding_distance(objectives, front)
                sorted_indices = np.argsort(distances)[::-1]
                for i in sorted_indices:
                    if len(new_population) < self.pop_size:
                        new_population.append(population[front[i]])
            else:
                new_population.extend(population[front])
        return np.array(new_population)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        for i in range(self.batch_size):
            if np.random.rand() < self.mutation_prob:
                individual[4 * i] = np.random.randint(0, self.n_path1 + 1)
                individual[4 * i + 1] = np.random.randint(0, self.n_path2 + 1)
                individual[4 * i + 2] = np.random.randint(1, self.n_cs + 1)
                individual[4 * i + 3] = np.random.randint(1, self.n_power + 1)
        return individual

    def run(self):
        population = self.initialize_population()
        for gen in range(self.n_gen):
            objectives = self.evaluate(population)
            fronts = self.non_dominated_sorting(objectives)
            selected_population = self.selection(population, fronts, objectives)
            new_population = []
            while len(new_population) < self.pop_size:
                parents = selected_population[np.random.choice(len(selected_population), 2, replace=False)]
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population.append(self.mutation(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutation(child2))
            population = np.array(new_population)
        return population