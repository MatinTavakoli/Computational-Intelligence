import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from matplotlib.lines import Line2D

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


class Network:
    def __init__(self, input_layer, hidden_layer, output_layer, activation_function):
        self.input_layer = input_layer  # number of neurons in the input layer
        self.hidden_layer = []  # number of neurons in the hidden layers (represented by an array of integers)
        self.output_layer = output_layer  # number of neurons in the output layer
        self.weights = []
        self.biases = []
        self.activation_function = activation_function

    def fill_weights_and_biases(self):
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            self.fill_weights_and_biases_without_hidden_layer()
        else:
            self.fill_weights_and_biases_with_hidden_layer()

    def fill_weights_and_biases_without_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        # creates a matrix of weights, corresponding to two adjacent layers
        weight_matrix = np.random.normal(0, 1, (n_i, n_o))
        self.weights.append(weight_matrix)
        # creates a vector of biases, corresponding to two adjacent layers
        bias_vector = np.random.normal(0, 1, (n_o, 1))
        self.biases.append(bias_vector)

    def fill_weights_and_biases_with_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        h = self.hidden_layer
        for i in range(len(self.hidden_layer) + 1):
            if i != len(self.hidden_layer):
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (n_i, h[i]))
                self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (h[i], 1))
                self.biases.append(bias_vector)
            else:
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (h[i], n_o))
                self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (n_o, 1))
                self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    # determine the values of the next layer, based on the values of the current layer
    def feed_forward(self, a, i):
        for b, w in zip(self.biases[i], self.weights[i]):
            if self.activation_function == "sigmoid":
                a = self.sigmoid(np.dot(w, a) + b)
        return a

    def forward_propagation_without_hidden_layer(self, a):
        return self.feed_forward(a, 0)

    def forward_propagation_with_hidden_layer(self, a):
        for i in range(len(self.hidden_layer) + 1):
            a = self.feed_forward(a, i)
        return a

    def plot_results(self, points, a):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='c1 (red)', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='c2 (blue)', markerfacecolor='b', markersize=10),
        ]
        fig, ax = plt.subplots()
        for point in points:
            max = 0
            best_i = 0
            for i in range(len(a)):
                if a[i] >= max:  # class 1: red
                    max = a[i]
                    best_i = i
            plt.scatter(point[0], point[1], color=colors[best_i])
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.legend(handles=legend_elements, loc='lower right')
        plt.show()


def main():
    points = []
    print('Welcome to the Artificial Neural Network Classifier!')
    time.sleep(0.8)
    with open('dataset.csv') as csv_file:
        csv_file.readline()  # line 0 is only metadata
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            points.append((float(line[0]), float(line[1]), float(line[2])))
    # separating train data with test data
    train_data = points[:300]  # the first 3/4 of the data is considered as train data
    test_data = points[300:]  # the final 1/4 of the data is considered as test data
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='c1 (red)', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='c2 (blue)', markerfacecolor='b', markersize=10),
    ]
    fig, ax = plt.subplots()
    for i in range(len(train_data)):
        if points[i][2] == 0.0:  # class 1: red
            plt.scatter(points[i][0], points[i][1], color='red')
        elif points[i][2] == 1.0:  # class 2: blue
            plt.scatter(points[i][0], points[i][1], color='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.legend(handles=legend_elements, loc='lower right')
    plt.title('train data')
    plt.show()

    fig, ax = plt.subplots()
    for i in range(len(test_data)):
        if points[i][2] == 0.0:  # class 1: red
            plt.scatter(points[i][0], points[i][1], color='red')
        elif points[i][2] == 1.0:  # class 2: blue
            plt.scatter(points[i][0], points[i][1], color='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.legend(handles=legend_elements, loc='lower right')
    plt.title('test data')
    plt.show()


if __name__ == "__main__":
    main()
