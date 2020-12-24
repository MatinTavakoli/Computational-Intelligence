import time
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.neighbor_edges = []


class steiner_tree:
    def __init__(self, steiner, terminal, edges, bools):
        self.steiner = steiner
        self.terminal = terminal
        self.vertices = steiner.copy()
        self.vertices.extend(terminal.copy())
        self.edges = edges
        self.bools = bools

        weights = []
        for i, edge in enumerate(edges):
            if self.bools[i]:
                start = edge[0]
                end = edge[1]
                delta_y = self.vertices[end].y - self.vertices[start].y
                delta_x = self.vertices[end].x - self.vertices[start].x
                distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
                weights.append(distance)
            else:
                weights.append(0)
        self.weights = weights


# def find_neighbors(chrom):
#     for i, edge in enumerate(chrom.edges):
#         if chrom.bools[i]:
#             start = edge[0]
#             end = edge[1]
#             chrom.vertices[start].neighbor_edges.append(i)
#             chrom.vertices[end].neighbor_edges.append(i)

def plot_graph(tree, s_color='g', t_color='r'):
    s_x = [node.x for node in tree.steiner]
    s_y = [node.y for node in tree.steiner]
    t_x = [node.x for node in tree.terminal]
    t_y = [node.y for node in tree.terminal]
    edges = tree.edges

    # Plot points
    steiner = plt.scatter(s_x, s_y, c=s_color)
    terminal = plt.scatter(t_x, t_y, c=t_color)

    x_nodes = s_x.copy()
    x_nodes.extend(t_x)
    y_nodes = s_y.copy()
    y_nodes.extend(t_y)

    for k, (i, j) in enumerate(edges):
        if tree.bools[k]:
            start = [x_nodes[i], x_nodes[j]]
            end = [y_nodes[i], y_nodes[j]]
            plt.plot(start, end, 'blue')

    plt.legend([steiner, terminal], ["Steiner nodes", "Terminal nodes"])

    plt.show()


def plot_result(size, min_coords, max_coords, avg_coords, min_color='r', max_color='b', avg_color='g'):
    # Plot points
    mins, = plt.plot(range(size), min_coords, c=min_color, label='best')
    maxs, = plt.plot(range(size), max_coords, c=max_color, label='worst')
    avgs, = plt.plot(range(size), avg_coords, c=avg_color, label='average')

    plt.legend(handles=[mins, maxs, avgs])

    plt.show()


def generate_random_chromosome(tree):
    new_bools = []
    for i in range(len(tree.edges)):
        p = random.random()
        if p > 0.5:
            new_bools.append(True)
        else:
            new_bools.append(False)
    return steiner_tree(tree.steiner, tree.terminal, tree.edges, new_bools)


def select_from_population(pop):
    return random.choice(pop)


# crossover
def generate_new_child(p1, p2):
    child_bools = [a or b for a, b in zip(p1.bools, p2.bools)]
    child_tree = steiner_tree(p1.steiner, p1.terminal, p1.edges, child_bools)
    return child_tree


def mutated_child(c, bool_mute_prob):
    # new_child_bools = []
    # for i in range(len(c.bools)):
    #     p = random.random()
    #     if p < bool_mute_prob:
    #         new_child_bools.append(not c.bools[i])
    #     else:
    #         new_child_bools.append(c.bools[i])
    # new_child_tree = steiner_tree(c.steiner, c.terminal, c.edges, new_child_bools)
    # return new_child_tree

    new_child_bools = c.bools
    t = random.choice(c.terminal)
    for edge_index in t.neighbor_edges:
        if not c.bools[edge_index]:
            new_child_bools[edge_index] = True
            break
    new_child_tree = steiner_tree(c.steiner, c.terminal, c.edges, new_child_bools)
    return new_child_tree


def evaluate_fitness(chrom):
    # # find_neighbors(chrom)
    # for i, terminal in enumerate(chrom.terminal):
    #     is_connected = False
    #     t_index = len(chrom.steiner) + i
    #     # for edge_index in chrom.vertices[t_index].neighbor_edges:
    #     for j, edge in enumerate(chrom.edges):
    #         if chrom.edges[0] == t_index or chrom.edges[1] == t_index:
    #             if not chrom.bools[j]:
    #                 chrom.bools[j] = True
    return 1 / sum(chrom.weights)


def main():
    print('Welcome to the Steiner Tree Problem Solver!')
    # time.sleep(1)
    print('Reading from input file...')
    # time.sleep(1)
    f = open("steiner_in.txt", "r")
    # number of steiner nodes, number of terminal nodes and number of edges respectively
    n_s, n_t, n_e = map(int, f.readline().split())

    s_x = []
    s_y = []

    t_x = []
    t_y = []

    edges = []

    for i, line in enumerate(f):
        # steiner nodes
        if i < n_s:
            line = line.replace('\n', '')
            x, y = map(int, line.split())
            s_x.append(x)
            s_y.append(y)
        # terminal nodes
        elif i < n_s + n_t:
            line = line.replace('\n', '')
            x, y = map(int, line.split())
            t_x.append(x)
            t_y.append(y)
        # edges
        elif i < n_s + n_t + n_e:
            line = line.replace('\n', '')
            edge_start, edge_end = map(int, line.split())
            edges.append([edge_start, edge_end])
    f.close()
    steiner = []
    terminal = []

    [steiner.append(node(x, y)) for x, y in zip(s_x, s_y)]
    [terminal.append(node(x, y)) for x, y in zip(t_x, t_y)]
    bools = []
    for i in range(len(edges)):
        bools.append(True)
    tree = steiner_tree(steiner, terminal, edges, bools)
    plot_graph(tree)

    generations_size = 50
    init_pop_size = 2000
    mutation_prob = 0.2
    bool_mutate_prob = 0.35

    # create initial population
    init_pop = []
    init_fitness = []
    for i in range(init_pop_size):
        new_tree = generate_random_chromosome(tree)
        init_pop.append(new_tree)
        # plot_graph(new_tree)

    pop = init_pop.copy()

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
        for j in range(init_pop_size):
            parent_1 = select_from_population(pop)
            parent_2 = select_from_population(pop)
            new_child_1 = generate_new_child(parent_1, parent_2)
            new_child_2 = generate_new_child(parent_1, parent_2)
            prob1 = random.random()
            if prob1 < mutation_prob:
                prob2 = random.random()
                if prob2 < 0.5:
                    new_child_1 = mutated_child(new_child_1, bool_mutate_prob)
                else:
                    new_child_2 = mutated_child(new_child_2, bool_mutate_prob)
            pop.append(new_child_1)
            pop.append(new_child_2)

        # evaluate fitness
        fitness_arr = []
        for j in range(len(pop)):
            fitness = evaluate_fitness(pop[j])
            fitness_arr.append(fitness)

        # selection
        next_pop = []
        next_fit = []
        for j in range(init_pop_size):
            best_fit = fitness_arr[0]
            best_gen = pop[0]
            for k in range(len(pop)):
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
        for j, chrom in enumerate(next_pop):
            f = 1 / fitness_arr[i]
            func_arr.append(f)

        min_func = min(func_arr)
        min_funcs.append(min_func)

        max_func = max(func_arr)
        max_funcs.append(max_func)

        avg_func = sum(func_arr) / init_pop_size
        avg_funcs.append(avg_func)

        print('min, max, avg value of function are: {}, {}, {}'.format(min_func, max_func, avg_func))

    print(max(fitness_arr), max(func_arr))
    plot_graph(pop[0])
    plot_result(generations_size, min_scores, max_scores, avg_scores)
    plot_result(generations_size, min_funcs, max_funcs, avg_funcs)

    # writing to file
    f = open("steiner_out.txt", "w")
    for i in range(len(pop[0].edges)):
        if pop[0].bools[i]:
            f.write(str(i) + '\n')
    f.write(str(1 / evaluate_fitness(pop[0])) + '\n')
    f.close()


if __name__ == "__main__":
    main()
