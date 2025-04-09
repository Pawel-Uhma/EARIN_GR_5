import random
import numpy as np
from functions import *

def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        
    def initialize_population(self):
        x_range, y_range = init_ranges[styblinski_tang_2d]
        population = [
            (random.uniform(*x_range), random.uniform(*y_range))
            for x in range(self.population_size)
        ]
        return population

    def evaluate_population(self, population):
        fitness = []
        for x in population:
            fitness.append(styblinski_tang_2d(*x))
        return fitness


    def selection(self, population, fitness_values):
        # sort indices of the population by fitness (lowest fitness is best)
        sorted_indices = np.argsort(fitness_values)
        max_weight = len(population)
        #creating array of length max weights
        weights = [0] * max_weight
        for rank, idx in enumerate(sorted_indices):
            weights[idx] = max_weight - rank
        # normalize
        weights = np.array(weights)
        probabilities = weights / np.sum(weights)
        # how many are selected for reproduction
        num_selected = int(self.population_size * self.crossover_rate)
        # randomly select
        selected_indices = np.random.choice(max_weight, size=num_selected, replace=True, p=probabilities)
        return [population[i] for i in selected_indices]


    def crossover(self, parents):
        np.random.shuffle(parents)
        children = []
        num_parents = len(parents)
        # if there are odd parents, duplicate the random parent
        if num_parents % 2:
            random_index = np.random.randint(num_parents)
            random_parent = parents[random_index]
            parents = np.concatenate([parents, [random_parent]])
            num_parents += 1
        for i in range(0, num_parents, 2):
            p1 = np.array(parents[i])
            p2 =  np.array(parents[i + 1])
            alpha = np.random.rand() 
            # child = parent1 * random(x) + parent2 * (1-random(x)) 
            child = alpha * p1 + (1 - alpha) * p2
            children.append(child)
        return np.array(children)


    def mutate(self, individuals):
        for i in range(len(individuals)):
            #for each x and y
            for j in range(2):  
                # decide by random if a child is mutated
                if np.random.rand() < self.mutation_rate:
                    # a random gaussian mutation to a child
                    individuals[i, j] += np.random.normal(0, self.mutation_strength)
                    # don't let the values out of range
                    x_range, y_range = init_ranges[styblinski_tang_2d]
                    if j == 0:
                        individuals[i, j] = np.clip(individuals[i, j], x_range[0], x_range[1])
                    elif j == 1:
                        individuals[i, j] = np.clip(individuals[i, j], y_range[0], y_range[1])
        return individuals
    
    def evolve(self, seed: int) -> ...:
        set_seed(seed)
        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)
            
            best_idx = np.argmin(fitness_values)
            best_solutions.append(population[best_idx])
            best_fitness_values.append(fitness_values[best_idx])
            average_fitness_values.append(np.average(fitness_values))
            
            parents_for_reproduction = self.selection(population, fitness_values)
            children = self.crossover(parents_for_reproduction)
            children = self.mutate(children)
            
            indices = np.random.choice(range(len(population)), size=len(children), replace=False)
            for i, child in zip(indices, children):
                population[i] = child
            
        return best_solutions, best_fitness_values, average_fitness_values
