import numpy as np
from TNplus import ChargeStation, Vehicle, DispatchCenter, Edge, Node



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


# Example usage
spea = SPEA(pop_size=100, n_gen=200)
final_population = spea.run(n_var=10)
print(final_population)