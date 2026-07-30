import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict(self, x):
        net = np.dot(x, self.weights) + self.bias
        return 1 if net >= 0 else 0

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

p = Perceptron([1,1], -0.5)
