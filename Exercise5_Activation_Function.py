import numpy as np

net=np.linspace(-5,5,10)

def step(x):
    return np.where(x>=0,1,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

print("Net:",net)
print("Step:",step(net))
print("Sigmoid:",sigmoid(net))
print("ReLU:",relu(net))
