import matplotlib.pyplot as plt
import numpy as np

def plot_results(best_fitness, average_fitness, best_solutions, test_case_data=None):
    generations = np.arange(1, len(best_fitness) + 1)
    best_solutions = np.array(best_solutions)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    
    ax1.plot(generations, best_fitness, marker='o', label='Best Fitness')
    ax1.plot(generations, average_fitness, marker='x', label='Average Fitness')
    ax1.set_ylabel("Fitness Value")
    ax1.set_title("Fitness over Generations")
    ax1.legend(loc='upper center')
    ax1.grid(True)
    
    ax2.plot(generations, best_solutions[:, 0], marker='o', label='Best Solution x')
    ax2.plot(generations, best_solutions[:, 1], marker='x', label='Best Solution y')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Coordinate Value")
    ax2.set_title("Best Solution Evolution over Generations")
    ax2.legend(loc='upper center')
    ax2.grid(True)
    
    if test_case_data is not None:
        textstr = '\n'.join(f'{key}: {value}' for key, value in test_case_data.items())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.show()
