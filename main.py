import numpy as np
import json
import sys
import csv
import random

command_inputs = list(sys.argv)
input_data = command_inputs[1]
json_data = command_inputs[2]

with open(json_data) as hyperparams:
    params = json.load(hyperparams)
    learning_rate = params['learning rate']
    iterations = params['num iter']

x_vectors = []
y_labels = []
with open(input_data) as data:
    reader = csv.reader(data, delimiter=' ')
    for row in reader:
        x_data = list([1] + [float(r) for r in row[:-1]])
        x_vectors.append(x_data)
        y_labels.append(float(row[-1]))

x_matrix = np.matrix(x_vectors)
x_transpose = x_matrix.T
y_vector = np.matrix(y_labels).T

weights = np.linalg.inv(x_transpose*x_matrix) * x_transpose * y_vector
print(weights)
#TODO: write to output file

#TODO: stochastic gradient descent

def stoch_grad_descent(iters, learn_rate, x_data, y_data):
    # initialize weights vector as vector of zeros
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    w_i = np.zeros((x_data.shape[1], 1))

    iters = 500000

    while iters > 0:
        i = random.randint(0, x_data.shape[1])
        X = x_data[i].reshape(1, x_data.shape[1])
        Y = y_data[i].reshape(1,1)
        w_i = w_i - learn_rate*np.dot(X.T, (np.dot(X, w_i) - Y))
        iters -= 1
    return w_i

sgd = stoch_grad_descent(iterations, learning_rate, x_vectors, y_labels)
print(sgd)
