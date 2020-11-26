import numpy as np
import random as rd
from random import randint
import matplotlib.pyplot as plt


def generate_treasure(n_items: int):
    """
    Creates the knapsack list of items
    """
    item_number = np.arange(1, n_items+1)
    weight = np.random.randint(1, 15, size=n_items)
    value = np.random.randint(10, 750, size=n_items)
    print(item_number.shape)
    print('The list is as follows:')
    print('Item No.\tWeight\t\tValue')
    for i in range(item_number.shape[0]):
        print('{0}\t\t{1}\t\t{2}\n'.format(item_number[i], weight[i], value[i]))

    return (item_number, weight, value)


def generate_population(n_chromosomes: int, n_genes: int):
    """
    Will create the initial population of solutions

    n_chromosomes - the number solutions in the population
    n_genes - number of genes per chromosome
    """
    pop_size = (n_chromosomes, n_genes)

    print('Population size = {}'.format(pop_size))

    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)

    print('Initial population: \n{}'.format(initial_population))

    return initial_population


def calculate_fitness(weight: int,
                      value: int,
                      population,
                      threshold: int):
    """
    Calculates the fitness of the population
        1. It iterates over each candidate solution
        2. Sum over each set bit and get value
        3. Sum over each set bit and get weight
        4. if weight exceeds threshold = 0 otherwise sum_value
    """

    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        sum_value = np.sum(population[i] * value)
        sum_weight = np.sum(population[i] * weight)

        # fitness = sum_value if not above threshold
        fitness[i] = sum_value if sum_weight <= threshold else 0

    return fitness.astype(int)


def selection(fitness,
              num_parents,
              population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    # for every iteration we pick the fittest and add to parents
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        # add the fittest to population
        parents[i, :] = population[max_fitness_idx[0][0], :]
        # ugly, but to prevent from being selected again.
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings, crossover_rate=0.8):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    # crossover point is the middle point
    crossover_point = int(parents.shape[1]/2)
    i = 0
    while (i < num_offsprings):
        # pick parent to cross over 
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        # tails of each parent are swapped to get new off-springs
        offsprings[i, 0:crossover_point] = parents[parent1_index,
                                                   0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index,
                                                  crossover_point:]
        i += 1
    return offsprings


def mutation(offsprings, mutation_rate=0.4):
    """
    Goes through each offspring and flips a bit
    if the random number is below mutation rate
    """
    mutants = np.empty((offsprings.shape))
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1]-1)
        mutants[i, int_random_value] = 1 \
            if mutants[i, int_random_value] == 0 else 0

    return mutants


def optimise_solution(weight,
                      value,
                      population,
                      population_size,
                      num_generations,
                      threshold):
    # recording values for drawing
    parameters, fitness_history = [], []
    
    num_parents = int(population_size/2)
    num_offsprings = population_size - num_parents

    for i in range(num_generations):
        fitness = calculate_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print('Last generation: \n{}\n'.format(population))
    fitness_last_gen = calculate_fitness(weight, value, population, threshold)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history


def draw_graph(fitness_history, num_generations):
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(num_generations)), fitness_history_mean, label='Mean Fitness')
    plt.plot(list(range(num_generations)), fitness_history_max, label='Max Fitness')
    plt.legend()
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    print(np.asarray(fitness_history).shape)


def main(capacity=35, pop_size=8, n_items=10, num_generations=50):
    # generate the list of items to pick from
    item_number, weight, value = generate_treasure(n_items)

    # generate the initial population
    initial_population = generate_population(n_chromosomes=pop_size,
                                             n_genes=item_number.shape[0])
    parameters, fitness_history = \
        optimise_solution(weight=weight,
                          value=value,
                          population=initial_population,
                          population_size=pop_size,
                          num_generations=num_generations,
                          threshold=capacity)

    print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
    selected_items = item_number * parameters
    print('\nSelected items that will maximize the knapsack without breaking it:')
    for i in range(selected_items.shape[1]):
        if selected_items[0][i] != 0:
            print('{}\n'.format(selected_items[0][i]))

    draw_graph(fitness_history, num_generations)


if __name__ == '__main__':
    main()
