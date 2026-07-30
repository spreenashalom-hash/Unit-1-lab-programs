import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])

w=np.array([1,1])
b=-0.5

for x in X:
    y=1 if np.dot(x,w)+b>=0 else 0
    print(x,y)

print("Single perceptron cannot solve XOR.")
