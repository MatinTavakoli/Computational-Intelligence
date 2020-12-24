import time
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class Point:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2


def plot_graph(size, min_coords, max_coords, avg_coords, min_color='r', max_color='b', avg_color='g'):

    # Plot points
    mins, = plt.plot(range(size), min_coords, c=min_color, label='best')
    maxs, = plt.plot(range(size), max_coords, c=max_color, label='worst')
    avgs, = plt.plot(range(size), avg_coords, c=avg_color, label='average')

    plt.legend(handles=[mins, maxs, avgs])

    plt.show()


def generate_random_chromosome():
    x1 = np.random.uniform(-512.0, 512.0)
    x2 = np.random.uniform(-512.0, 512.0)
    return Point(x1, x2)


# crossover
def combine_from_population(pop, n):
    parents = []
    for i in range(n):
        parents.append(random.choice(pop))

    avg_x1 = 0
    avg_x2 = 0
    for i in range(n):
        avg_x1 += parents[i].x1
        avg_x2 += parents[i].x2
    avg_x1 /= n
    avg_x2 /= n
    return Point(avg_x1, avg_x2)


# mutation
def generate_children(p, n, mutation_prob, sigma, tau):
    children = []
    for i in range(n):
        prob1 = random.random()
        if prob1 < mutation_prob:
            norm = np.random.normal(0, 1)
            sigma = sigma * math.e ** (-tau * norm)
            children.append(Point(p.x1 + sigma, p.x2 + sigma))
        else:
            children.append(Point(p.x1, p.x2))
    return children


def mutated_child(c, bool_prob, sigma, tau):
    p = random.random()
    if p < bool_prob:
        norm = np.random.normal(0, 1)
        sigma = sigma * math.e ** (-tau * norm)
        return Point(c.x1 + sigma, c.x2 + sigma), sigma
    return c, sigma


def evaluate_fitness(chrom):
    if chrom.x1 < -512 or chrom.x1 > 512 or chrom.x2 < -512 or chrom.x2 > 512:
        return 0
    f = -1 * (chrom.x2 + 47) * math.sin(math.sqrt(abs(chrom.x2 + chrom.x1 / 2 + 47))) - chrom.x1 * math.sin(
        math.sqrt(abs(chrom.x1 - (chrom.x2 + 47))))
    global_min = -959.6407
    return 1 / abs(f - global_min)


def main():
    print('Welcome to the Eggholder Problem Solver!')
    # time.sleep(1)

    generations_size = 70
    init_pop_size = 3000
    mutation_prob = 0.2
    sigma = 10
    tau = 0.3
    parent_combine_factor = init_pop_size // 10

    # create initial population
    init_pop = []
    init_fitness = []
    for i in range(init_pop_size):
        new_point = generate_random_chromosome()
        init_pop.append(new_point)
        # plot_graph(new_point)

    pop = init_pop

    min_scores = []
    max_scores = []
    avg_scores = []

    min_funcs = []
    max_funcs = []
    avg_funcs = []

    # generations
    for i in range(generations_size):
        print('generation #{}'.format(i))
        # create children
        parent = combine_from_population(pop, parent_combine_factor)
        children = generate_children(parent, init_pop_size, mutation_prob, sigma, tau)
        pop.extend(children)

        # evaluate fitness
        fitness_arr = []
        for j in range(len(pop)):
            fitness = evaluate_fitness(pop[j])
            fitness_arr.append(fitness)

        # selection
        next_pop = []
        next_fit = []
        for j in range(init_pop_size):
            best_fit = fitness_arr[len(pop) // 2]
            best_gen = pop[len(pop) // 2]
            for k in range(len(pop) // 2, len(pop)):
                if fitness_arr[k] > best_fit:
                    best_fit = fitness_arr[k]
                    best_gen = pop[k]
            next_pop.append(best_gen)
            next_fit.append(best_fit)
            pop.remove(best_gen)
            fitness_arr.remove(best_fit)
        pop = next_pop
        fitness_arr = next_fit

        min_score = min(fitness_arr)
        min_scores.append(min_score)

        max_score = max(fitness_arr)
        max_scores.append(max_score)

        avg_score = sum(fitness_arr) / init_pop_size
        avg_scores.append(avg_score)

        print('min, max, avg score are: {}, {}, {}'.format(min_score, max_score, avg_score))

        func_arr = []
        for chrom in next_pop:
            f = -1 * (chrom.x2 + 47) * math.sin(math.sqrt(abs(chrom.x2 + chrom.x1 / 2 + 47))) - chrom.x1 * math.sin(
                math.sqrt(abs(chrom.x1 - (chrom.x2 + 47))))
            func_arr.append(f)

        min_func = min(func_arr)
        min_funcs.append(min_func)

        max_func = max(func_arr)
        max_funcs.append(max_func)

        avg_func = sum(func_arr) / init_pop_size
        avg_funcs.append(avg_func)

        print('min, max, avg value of function are: {}, {}, {}'.format(min_func, max_func, avg_func))

    print(max(fitness_arr))
    # final_x1 = [chrom.x1 for chrom in pop if evaluate_fitness(chrom) != 0]
    # final_x2 = [chrom.x2 for chrom in pop if evaluate_fitness(chrom) != 0]
    plot_graph(generations_size, min_scores, max_scores, avg_scores)
    plot_graph(generations_size, min_funcs, max_funcs, avg_funcs)


if __name__ == "__main__":
    main()
