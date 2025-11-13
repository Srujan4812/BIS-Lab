import numpy as np
import matplotlib.pyplot as plt

# Objective Function (Rastrigin)
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Initialize population grid
def initialize_population(rows, cols, dim, lb, ub):
    return np.random.uniform(lb, ub, size=(rows, cols, dim))

# Fitness evaluation
def evaluate_population(pop):
    rows, cols, _ = pop.shape
    fitness = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            fitness[i, j] = rastrigin(pop[i, j])
    return fitness

# Get Von Neumann neighborhood
def get_neighbors(i, j, rows, cols):
    return [
        ((i - 1) % rows, j),
        ((i + 1) % rows, j),
        (i, (j - 1) % cols),
        (i, (j + 1) % cols)
    ]

# Tournament selection among neighbors
def local_selection(pop, fitness, i, j):
    neighbors = get_neighbors(i, j, *fitness.shape)
    candidates = [(i, j)] + neighbors
    best = min(candidates, key=lambda x: fitness[x])
    return pop[best].copy()

# Simulated Binary Crossover (SBX)
def crossover(parent1, parent2, eta=2):
    child1, child2 = parent1.copy(), parent2.copy()
    for k in range(len(parent1)):
        if np.random.rand() < 0.5:
            beta = (2 * np.random.rand()) ** (1 / (eta + 1))
            child1[k] = 0.5 * ((1 + beta) * parent1[k] + (1 - beta) * parent2[k])
            child2[k] = 0.5 * ((1 - beta) * parent1[k] + (1 + beta) * parent2[k])
    return child1, child2

# Mutation
def mutate(individual, lb, ub, mutation_rate=0.1):
    for k in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[k] += np.random.uniform(-0.1, 0.1)
            individual[k] = np.clip(individual[k], lb, ub)
    return individual

# Parallel Cellular Genetic Algorithm
def cellular_ga(rows=5, cols=5, dim=5, lb=-5.12, ub=5.12, generations=200):
    pop = initialize_population(rows, cols, dim, lb, ub)
    fitness = evaluate_population(pop)
    best_history = []

    for gen in range(generations):
        new_pop = pop.copy()

        for i in range(rows):
            for j in range(cols):
                p1 = local_selection(pop, fitness, i, j)
                p2 = local_selection(pop, fitness, i, j)

                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, lb, ub)
                c2 = mutate(c2, lb, ub)

                # Pick better child
                f1, f2 = rastrigin(c1), rastrigin(c2)
                if f1 < fitness[i, j] or f2 < fitness[i, j]:
                    if f1 < f2:
                        new_pop[i, j] = c1
                        fitness[i, j] = f1
                    else:
                        new_pop[i, j] = c2
                        fitness[i, j] = f2

        pop = new_pop
        best_fitness = np.min(fitness)
        best_history.append(best_fitness)

        if gen % 20 == 0 or gen == generations - 1:
            print(f"Generation {gen:3d} | Best Fitness: {best_fitness:.6e}")

    best_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
    best_solution = pop[best_idx]
    return best_solution, np.min(fitness), best_history


# Run the algorithm
if __name__ == "__main__":
    np.random.seed(42)
    best_sol, best_val, history = cellular_ga(rows=6, cols=6, dim=5, generations=200)

    print("\n=== FINAL RESULT ===")
    print("Best Solution:", np.round(best_sol, 6))
    print("Best Fitness:", best_val)

    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Parallel Cellular Genetic Algorithm (Rastrigin Function)")
    plt.grid(True)
    plt.show()
