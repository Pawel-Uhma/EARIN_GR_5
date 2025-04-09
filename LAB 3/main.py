from genetic_algorithm import GeneticAlgorithm
from plotting import plot_results
from test_cases import crossover_tests, mutation_rate_tests, mutation_strength_tests

def run_test_case(test_id, population_size, mutation_rate, mutation_strength, crossover_rate, num_generations, seed):
    print(f"\n{'='*40}\nRunning test case {test_id}")
    print(f"Population Size: {population_size}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Mutation Strength: {mutation_strength}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Number of Generations: {num_generations}")
    print(f"Seed: {seed}\n{'='*40}")

    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        crossover_rate=crossover_rate,
        num_generations=num_generations,
    )

    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=seed)
    print("Best solution found:", best_solutions[-1])
    print("Best fitness value:", best_fitness_values[-1])
    return best_solutions, best_fitness_values, average_fitness_values

def main():
    # MODIFY THIS PART FOR DIFFERENT TEST CASES 
    # choose: crossover_tests, mutation_rate_tests, mutation_strength_tests
    test_cases_group = crossover_tests

    for tc in test_cases_group["test_cases"]:
        best_solutions, best_fitness_values, average_fitness_values = run_test_case(
            test_id=tc["test_id"],
            population_size=tc["population_size"],
            mutation_rate=tc["mutation_rate"],
            mutation_strength=tc["mutation_strength"],
            crossover_rate=tc["crossover_rate"],
            num_generations=tc["num_generations"],
            seed=tc["seed"]
        )
        plot_results(best_fitness_values, average_fitness_values, best_solutions, test_case_data=tc)
    
if __name__ == "__main__":
    main()
