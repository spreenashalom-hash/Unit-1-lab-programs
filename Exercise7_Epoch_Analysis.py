import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([0,0,0,1])

w=np.zeros(2)
b=0
lr=0.2

epochs=20

for epoch in range(epochs):

    total_error=0

    for x,target in zip(X,Y):

        net=np.dot(x,w)+b
        out=1 if net>=0 else 0

        error=target-out

        w=w+lr*error*x
        b=b+lr*error

        total_error+=abs(error)

    print("Epoch",epoch+1,"Error",total_error)

    if total_error==0:
        print("Converged")
        break
