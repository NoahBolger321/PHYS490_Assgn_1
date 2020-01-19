import numpy as np
import json
import sys
import csv

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

