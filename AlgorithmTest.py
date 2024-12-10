# The `pymoo.algorithms.moo` module in the `pymoo` library includes several multi-objective optimization algorithms. Here are some of the algorithms available:
#
# - `NSGA2`: Non-dominated Sorting Genetic Algorithm II
# - `NSGA3`: Non-dominated Sorting Genetic Algorithm III
# - `SPEA2`: Strength Pareto Evolutionary Algorithm 2
# - `MOEAD`: Multi-Objective Evolutionary Algorithm based on Decomposition
# - `RVEA`: Reference Vector Guided Evolutionary Algorithm
# - `CMAES`: Covariance Matrix Adaptation Evolution Strategy for multi-objective optimization
# - `UNSGA3`: Unified NSGA-III
#
# These algorithms are designed to handle multi-objective optimization problems and can be used to find a set of Pareto-optimal solutions.


'''
调度目标：充电车辆是否充电、充电站的选择、充电功率的选择、充电车辆的路径规划（起点-充电站-终点）
调度状态：路网形状、每条路车数、扩展的话加上车辆位置（road, next_road, distance, iswait）
G, road.capacity['all'], vehicle.road, vehicle.next_road, vehicle.distance, vehicle.iswait
调度约束：电量要够到充电、充电站容量不能超、一车至多对应一桩（反之亦然）、分配时需保证充电站仍足够满足该车充电需求
调度模板算法：多目标优化算法（NSGA2、SPEA2等）
'''

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultTermination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from TNplus import ChargeStation, Vehicle, DispatchCenter, Edge, Node

class ChargingStationProblem(Problem):
    def __init__(self, vehicles, path_results, cs):
        super().__init__(n_var=len(cs) * len(vehicles), n_obj=2, n_constr=1, xl=0, xu=1, type_var=np.int)
        self.vehicles = vehicles
        self.path_results = path_results
        self.cs = cs

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum(x)  # Objective 1: Minimize the number of charging stations used
        f2 = np.sum([self.calculate_cost(i, j) for i, j in zip(x[:-1], x[1:])])  # Objective 2: Minimize the total cost
        out["F"] = [f1, f2]

        # Example constraint: sum of variables should be less than or equal to a threshold
        g1 = np.sum(x) - 5  # Constraint 1: sum of x should be <= 5
        out["G"] = [g1]

    def calculate_cost(self, current_node, next_node):
        # Implement the cost calculation based on your specific requirements
        return np.random.rand()  # Placeholder for actual cost calculation



def assign_cs_nsga2(vehicles, path_results, cs, pop_size=100, n_gen=200):
    problem = ChargingStationProblem(vehicles, path_results, cs)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    termination = DefaultTermination(n_max_gen=n_gen)

    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

    best_solution = res.X[np.argmin(res.F[:, 1])]  # Select the solution with the minimum cost

    # Update each vehicle's path and charging station assignment
    for i, vehicle in enumerate(vehicles):
        vehicle_solution = best_solution[i * len(cs):(i + 1) * len(cs)]
        vehicle.path = vehicle_solution
        vehicle.road = vehicle_solution[0]
        if len(vehicle_solution) > 1:
            vehicle.next_road = vehicle_solution[1]
        else:
            vehicle.next_road = -1

        charge_id = vehicle_solution[-1]
        vehicle.charge = (charge_id, list(cs[charge_id].pile.keys())[np.random.randint(0, len(list(cs[charge_id].pile.keys())))])

        if vehicle.log:
            print(f"Vehicle {vehicle.id} assigned to charging station {charge_id} with power {vehicle.charge[1]}")



def assign_cs_spea2(vehicles, path_results, cs, pop_size=100, n_gen=200):
    problem = ChargingStationProblem(vehicles, path_results, cs)

    algorithm = SPEA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    termination = DefaultTermination(n_max_gen=n_gen)

    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

    best_solution = res.X[np.argmin(res.F[:, 1])]  # Select the solution with the minimum cost

    # Update each vehicle's path and charging station assignment
    for i, vehicle in enumerate(vehicles):
        vehicle_solution = best_solution[i * len(cs):(i + 1) * len(cs)]
        vehicle.path = vehicle_solution
        vehicle.road = vehicle_solution[0]
        if len(vehicle_solution) > 1:
            vehicle.next_road = vehicle_solution[1]
        else:
            vehicle.next_road = -1

        charge_id = vehicle_solution[-1]
        vehicle.charge = (charge_id, list(cs[charge_id].pile.keys())[np.random.randint(0, len(list(cs[charge_id].pile.keys())))])

        if vehicle.log:
            print(f"Vehicle {vehicle.id} assigned to charging station {charge_id} with power {vehicle.charge[1]}")






import numpy as np

class SPEA:
    def __init__(self, pop_size, n_gen, crossover_prob=0.9, mutation_prob=0.1):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self, n_var):
        return np.random.randint(0, 2, (self.pop_size, n_var))

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

# Example usage
spea = SPEA(pop_size=100, n_gen=200)
final_population = spea.run(n_var=10)
print(final_population)