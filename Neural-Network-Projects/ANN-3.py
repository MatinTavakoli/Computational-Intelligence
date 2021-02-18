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
        self.hidden_layer = hidden_layer  # number of neurons in the hidden layers (represented by an array of integers)
        self.output_layer = output_layer  # number of neurons in the output layer
        self.activation_function = activation_function
        self.weights = []
        self.biases = []
        self.neuron_values = []

    def fill_weights_and_biases(self):
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            self.fill_weights_and_biases_without_hidden_layer()
        else:
            self.fill_weights_and_biases_with_hidden_layer()

    def fill_weights_and_biases_without_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        # creates a matrix of weights, corresponding to two adjacent layers
        weight_matrix = np.random.normal(0, 1, (n_o, n_i))
        self.weights.append(weight_matrix)
        # creates a vector of biases, corresponding to two adjacent layers
        bias_vector = np.random.normal(0, 1, (n_o, 1))
        self.biases.append(bias_vector)

    def fill_weights_and_biases_with_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        h = self.hidden_layer
        for i in range(len(self.hidden_layer) + 1):
            if i == 0:
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (h[i], n_i))
                self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (h[i], 1))
                self.biases.append(bias_vector)
            elif i != len(self.hidden_layer):
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (h[i], h[i - 1]))
                self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (h[i], 1))
                self.biases.append(bias_vector)
            else:
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (n_o, h[i - 1]))
                self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (n_o, 1))
                self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def sigmoid_prime(self, x):
        return x * (1 - x)

    # determine the values of the next layer, based on the values of the current layer
    def feed_forward(self, a, i):
        if i == 0:
            self.neuron_values.append(a)  # appending the values of the initialized input neurons
        if self.activation_function == "sigmoid":
            w = self.weights[i]
            b = self.biases[i]
            a = self.sigmoid(np.dot(w, a) + b)
        self.neuron_values.append(a)
        return a

    def forward_propagation(self, a):
        self.neuron_values = []  # reset
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            return self.forward_propagation_without_hidden_layer(a)
        else:
            return self.forward_propagation_with_hidden_layer(a)

    def forward_propagation_without_hidden_layer(self, a):
        return self.feed_forward(a, 0)

    def forward_propagation_with_hidden_layer(self, a):
        for i in range(len(self.hidden_layer) + 1):
            a = self.feed_forward(a, i)
        return a

    def feed_backward(self, a, data, i, a_gradients=None):
        w_gradients = np.zeros(self.weights[i - 1].shape)
        b_gradients = np.zeros(self.biases[i - 1].shape)
        new_a_gradients = np.zeros((len(a[i - 1]), 1))  # for the next round
        for j in range(len(a[i])):
            if a_gradients is not None:
                b_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j])
                b_gradients[j] = b_gradient
            else:
                b_error = 2 * (a[i][j] - data[2 + j])  # only for the last layer
                b_gradient = b_error * self.sigmoid_prime(a[i][j])
                b_gradients[j] = b_gradient
            for m in range(len(self.weights[i - 1][j])):
                if a_gradients is not None:
                    w_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j]) * a[i - 1][m]
                    a_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j]) * self.weights[i - 1][j][m]
                else:
                    w_error = 2 * (a[i][j] - data[2 + j])  # only for the last layer
                    w_gradient = w_error * self.sigmoid_prime(a[i][j]) * a[i - 1][m]
                    a_gradient = w_error * self.sigmoid_prime(a[i][j]) * self.weights[i - 1][j][m]
                new_a_gradients[m][0] += a_gradient
                w_gradients[j][m] = w_gradient
        if len(self.hidden_layer) == 0:
            gradients = w_gradients, b_gradients
        else:
            gradients = w_gradients, b_gradients, new_a_gradients
        return gradients

    def backward_propagation(self, data):
        a = self.neuron_values
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            return self.backward_propagation_without_hidden_layer(a, data)
        else:
            return self.backward_propagation_with_hidden_layer(a, data)

    def backward_propagation_without_hidden_layer(self, a, data):
        return self.feed_backward(a, data, 1)

    def backward_propagation_with_hidden_layer(self, a, data):
        layer_gradients = []
        a_gradients = None
        for i in range(len(self.hidden_layer) + 1, 0, -1):
            w_gradients, b_gradients, a_gradients = self.feed_backward(a, data, i, a_gradients)
            layer_gradients.append((w_gradients, b_gradients))
        return layer_gradients

    def update_weights_and_biases(self, data_gradients, n, learning_rate):
        a = len(self.neuron_values)
        for i in range(a - 1):
            w_matrix = np.zeros(self.weights[i].shape)
            for j in range(n):
                w_matrix += data_gradients[j][a - 1 - i - 1][0][0]
            w_matrix = w_matrix * learning_rate / n
            self.weights[i] -= w_matrix

            b_matrix = np.zeros(self.biases[i].shape)
            for j in range(n):
                b_matrix += data_gradients[j][a - 1 - i - 1][1][0]
            b_matrix = b_matrix * learning_rate / n
            self.biases[i] -= b_matrix

    def plot_single_point(self, point, a):
        # # multi neuron output
        # max = 0
        # best_i = 0
        # for i in range(len(a)):
        #     if a[i] >= max:
        #         max = a[i]
        #         best_i = i
        # plt.scatter(data[0], data[1], color=colors[best_i])

        # single neuron output
        if a[0] < 0.5:
            color = 'r'
        else:
            color = 'b'
        plt.scatter(point[0], point[1], color=color)

    def show_result(self, points, results):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='c1 (red)', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='c2 (blue)', markerfacecolor='b', markersize=10),
        ]
        fig, ax = plt.subplots()

        for point, result in zip(points, results):
            self.plot_single_point(point, result)

        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.legend(handles=legend_elements, loc='lower right')
        plt.show()

    def print_accuracy(self, test_data, results):
        score = 0
        for i, data in enumerate(test_data):
            if (results[i] >= 0.5 and data[2] == 1) or (results[i] < 0.5 and data[2] == 0):
                score = score + 1
        return score


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

    network = Network(2, [2], 1, "sigmoid")
    network.fill_weights_and_biases()

    learning_rate = 0.43

    error = []
    for i in range(600):
        data_gradients = []
        sum = 0
        for data in train_data:
            network.forward_propagation(np.reshape((data[0], data[1]), (2, 1)))
            sum += (network.neuron_values[1][0] - data[2]) ** 2
            layer_gradients = network.backward_propagation(data)
            data_gradients.append(layer_gradients)

        error.append(sum)

        network.update_weights_and_biases(data_gradients, len(train_data), learning_rate)

    plt.plot(range(len(error)), error)

    results = []
    for data in test_data:
        result = network.forward_propagation(np.reshape((data[0], data[1]), (2, 1)))
        results.append(result)

    print("accuracy is {}%".format(network.print_accuracy(test_data, results)))
    network.show_result(test_data, results)


if __name__ == "__main__":
    main()
