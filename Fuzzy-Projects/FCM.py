import csv
import matplotlib.pyplot as plt
import random
import time
from matplotlib.lines import Line2D
from numpy import log as ln

colors = [
    "r",
    "g",
    "b",
    "c",
    "m",
    "y",
    "k",
    "orange",
    "purple",
    "brown",
]


def generate_random_centroids(c, points):
    n = len(points)
    means = []
    for i in range(c):  # iterating over all clusters
        # partition = points[n * i // c: n * (i + 1) // c]
        # mean = (sum([point[0] for point in partition]) / (n / c), sum([point[1] for point in partition]) / (n / c))
        mean = (random.uniform(0, 1), random.uniform(0, 1))
        # plt.scatter(mean[0], mean[1])
        means.append(mean)
    # plt.show()
    return means


def generate_random_memberships(c, points):
    n = len(points)
    memberships = []
    for k in range(n):  # iterating over all points
        membership = []
        for i in range(c):  # iterating over all clusters
            val = random.uniform(0, 1)
            membership.append(val)
        memberships.append(membership)
    return memberships


def recompute_membership_values(c, m, points, means):
    n = len(points)
    mu_ik = []
    for k in range(n):  # iterating over all points
        mu_i = []
        for i in range(c):  # iterating over all clusters
            sum = 0
            numerator_x = abs(points[k][0] - means[i][0])
            numerator_y = abs(points[k][1] - means[i][1])
            numerator = (numerator_x ** 2 + numerator_y ** 2) ** (1 / 2)
            for j in range(c):  # iterating over all clusters

                denominator_x = abs(points[k][0] - means[j][0])
                denominator_y = abs(points[k][1] - means[j][1])
                denominator = (denominator_x ** 2 + denominator_y ** 2) ** (1 / 2)

                res = (numerator / denominator) ** (2 / (m - 1))
                sum += res

            sum = 1 / sum
            mu_i.append(sum)
        mu_ik.append(mu_i)
    return mu_ik


def recompute_centroids(c, m, points, memberships):
    n = len(points)
    means = []
    for i in range(c):  # iterating over all clusters

        numerator_x_sum = 0
        denominator_x_sum = 0

        numerator_y_sum = 0
        denominator_y_sum = 0

        for k in range(n):  # iterating over all points

            numerator_x = (memberships[k][i] ** m) * points[k][0]
            denominator_x = (memberships[k][i] ** m)
            numerator_x_sum += numerator_x
            denominator_x_sum += denominator_x

            numerator_y = (memberships[k][i] ** m) * points[k][1]
            denominator_y = (memberships[k][i] ** m)
            numerator_y_sum += numerator_y
            denominator_y_sum += denominator_y
        res_x = numerator_x_sum / denominator_x_sum
        res_y = numerator_y_sum / denominator_y_sum
        means.append((res_x, res_y))
    return means


def determine_best_c(points, c_means, c_memberships, c_start):
    n = len(points)
    entropy_vals = []
    for p in range(len(c_memberships)):
        c = c_start + p
        outer_sigma = 0
        for k in range(n):  # iterating over all points
            inner_sigma = 0
            for i in range(c):
                numerator = c_memberships[p][k][i]
                denominator = ln(c_memberships[p][k][i])
                res = numerator * denominator
                inner_sigma += res
            outer_sigma += inner_sigma
        val = outer_sigma * -1
        val = val / ln(c)
        entropy_vals.append(val)
    print("entropy vals for c from 2 to 10 are: {}".format(entropy_vals))
    min = entropy_vals[0]
    best_c = c_start
    best_membership = c_memberships[0]
    best_means = c_means[0]
    for i, c_val in enumerate(entropy_vals):
        if c_val < min:
            min = c_val
            best_c = c_start + i
            best_membership = c_memberships[i]
            best_means = c_means[i]

    print("the best c is: {}".format(best_c))
    return best_c, best_means, best_membership


def classify_points(c, points, means, memberships, colors):
    n = len(points)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='means', markerfacecolor='gold', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c1 (red)', markerfacecolor=colors[0], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c2 (green)', markerfacecolor=colors[1], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c3 (blue)', markerfacecolor=colors[2], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c4 (cyan)', markerfacecolor=colors[3], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c5 (magneta)', markerfacecolor=colors[4],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c6 (yellow)', markerfacecolor=colors[5],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c7 (black)', markerfacecolor=colors[6], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c8 (orange)', markerfacecolor=colors[7],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c9 (purple)', markerfacecolor=colors[8],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c10 (brown)', markerfacecolor=colors[9], markersize=10),
    ]
    fig, ax = plt.subplots()
    for k in range(n):  # iterating over all points
        max = 0
        index = -1
        for i in range(c):  # iterating over all clusters
            if memberships[k][i] > max:
                max = memberships[k][i]
                index = i
        plt.scatter(points[k][0], points[k][1], color=colors[index])
    for mean in means:
        plt.scatter(mean[0], mean[1], color="gold")
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.98, 1.0), loc='upper left')
    plt.show()


def compute_c_means(points):
    n = len(points)
    m = 3  # power coefficient
    c_memberships = []  # memberships associated to c
    c_means = []  # memberships associated to c
    for c in range(2, 11):
        means = generate_random_centroids(c, points)
        memberships = []
        for i in range(10):
            memberships = recompute_membership_values(c, m, points, means)
            means = recompute_centroids(c, m, points, memberships)
        c_memberships.append(memberships)
        c_means.append(means)
    c, means, memberships = determine_best_c(points, c_means, c_memberships, 2)
    classify_points(c, points, means, memberships, colors)  # plot 'em!


def main():
    points = []
    print('Welcome to the Fuzzy C-means Classifier!')
    time.sleep(0.8)
    print('Select a number from 1 to 5')
    n = int(input())
    filename = 'sample{}.csv'.format(n)
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            points.append((float(line[0]), float(line[1])))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='original data', markerfacecolor='black', markersize=10)]
    fig, ax = plt.subplots()
    for i in range(len(points)):
        plt.scatter(points[i][0], points[i][1], color='black')
    ax.legend(handles=legend_elements, loc='lower right')
    plt.show()

    compute_c_means(points)


if __name__ == "__main__":
    main()
