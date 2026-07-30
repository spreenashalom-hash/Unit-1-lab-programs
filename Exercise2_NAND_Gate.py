import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict(self, x):
        net = np.dot(x, self.weights) + self.bias
        return 1 if net >= 0 else 0

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

p = Perceptron([-1,-1], 1.5)

print("X1 X2 NAND")
for x in inputs:
    print(x[0], x[1], p.predict(x))
