import random
from genetic_algorithm import GeneticAlgorithm
from plotting import plot_results
from test_cases import crossover_tests, mutation_rate_tests, mutation_strength_tests, population_tests

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

    best_solutions, best_fitness_values, average_fitness_values, convergence_generation = ga.evolve(seed=seed)
    print("Best solution found:", best_solutions[-1])
    print("Best fitness value:", best_fitness_values[-1])
    if convergence_generation < num_generations:
        print("Convergence achieved at generation:", convergence_generation)
    else:
        print(f"Convergence was not close enough in {num_generations} generations")
    return best_solutions, best_fitness_values, average_fitness_values, convergence_generation

def run_random_test_series(num_iterations=100):
    base_test = crossover_tests["test_cases"][0]
    best_conv = None
    best_params = None
    best_results = None

    for i in range(num_iterations):
        random_test_case = {
            "test_id": f"Random_{i+1}",
            "population_size": base_test["population_size"],
            "mutation_rate": round(random.uniform(0.1, 0.9), 1),
            "mutation_strength": round(random.uniform(0.1, 0.9), 1),
            "crossover_rate": round(random.uniform(0.1, 0.9), 1),
            "num_generations": base_test["num_generations"],
            "seed": random.randint(1, 1000)
        }
        
        bs, bf, af, conv_gen = run_test_case(
            test_id=random_test_case["test_id"],
            population_size=random_test_case["population_size"],
            mutation_rate=random_test_case["mutation_rate"],
            mutation_strength=random_test_case["mutation_strength"],
            crossover_rate=random_test_case["crossover_rate"],
            num_generations=random_test_case["num_generations"],
            seed=random_test_case["seed"]
        )
        
        if best_conv is None or conv_gen < best_conv:
            best_conv = conv_gen
            best_params = random_test_case
            best_results = (bs, bf, af)
            
    print(f"\nBest random test case converged at generation: {best_conv}")
    print("Best parameters found:", best_params)
    plot_results(best_results[1], best_results[2], best_results[0], test_case_data=best_params)
    return best_params, best_results, best_conv

def main():
    # MODIFY THIS PART FOR DIFFERENT TEST CASES 
    # choose: crossover_tests, mutation_rate_tests, mutation_strength_tests, population_tests
    test_cases_group = population_tests

    for tc in test_cases_group["test_cases"]:
        bs, bf, af, conv_gen = run_test_case(
            test_id=tc["test_id"],
            population_size=tc["population_size"],
            mutation_rate=tc["mutation_rate"],
            mutation_strength=tc["mutation_strength"],
            crossover_rate=tc["crossover_rate"],
            num_generations=tc["num_generations"],
            seed=tc["seed"]
        )
        tc["convergence_generation"] = conv_gen
        plot_results(bf, af, bs, test_case_data=tc)
    
    # run_random_test_series(num_iterations=100)

if __name__ == "__main__":
    main()
