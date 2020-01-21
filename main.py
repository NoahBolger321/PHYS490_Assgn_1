import numpy as np
import json
import sys
import csv
import random

# parse command line params
command_inputs = list(sys.argv)
input_data = command_inputs[1]
json_data = command_inputs[2]

# retrieve json data
with open(json_data) as hyperparams:
    params = json.load(hyperparams)
    learning_rate = params['learning rate']
    iterations = params['num iter']

# retrieve training data
x_vectors = []
y_labels = []
with open(input_data) as data:
    reader = csv.reader(data, delimiter=' ')
    for row in reader:
        x_data = list([1] + [float(r) for r in row[:-1]])
        x_vectors.append(x_data)
        y_labels.append(float(row[-1]))

# vector and matrix handling
x_matrix = np.matrix(x_vectors)
x_transpose = x_matrix.T
y_vector = np.matrix(y_labels).T

# retrieve the resulting weights of least squares regression
weights = np.linalg.inv(x_transpose*x_matrix) * x_transpose * y_vector

def stoch_grad_descent(iters, learn_rate, x_data, y_data):
    # initialize weights vector as vector of zeros
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    w_i = np.zeros((x_data.shape[1], 1))

    # change iterations
    iters = 500000

    while iters > 0:
        i = random.randint(0, x_data.shape[1])
        X = x_data[i].reshape(1, x_data.shape[1])
        Y = y_data[i].reshape(1,1)
        w_i = w_i - learn_rate*np.dot(X.T, (np.dot(X, w_i) - Y))
        iters -= 1
    return w_i

# retrieve the results of stochastic gradient descent
sgd = stoch_grad_descent(iterations, learning_rate, x_vectors, y_labels)

# open and write analytic/sgw results to output file
with open(input_data.replace('.in', '.out'), 'w') as f:
    for i in range(0, weights.size):
        f.write(str(float(weights[i])))
        f.write('\n')

    f.write('\n')

    for j in range(0, sgd.size):
        f.write(str(float(sgd[j])))
        f.write('\n')
