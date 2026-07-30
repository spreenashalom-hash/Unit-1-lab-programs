import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([0,0,0,1])

w=np.array([0.3,0.1])
b=0
lr=0.2

for x,target in zip(X,Y):

    net=np.dot(x,w)+b
    output=1 if net>=0 else 0

    error=target-output

    w=w+lr*error*x
    b=b+lr*error

    print("Input:",x)
    print("Net:",net)
    print("Output:",output)
    print("Error:",error)
    print("Weights:",w)
    print("Bias:",b)
