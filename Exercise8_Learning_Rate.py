import numpy as np

learning_rates=[0.1,0.5,1.0]

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([0,0,0,1])

for lr in learning_rates:

    w=np.zeros(2)
    b=0

    for epoch in range(50):

        total=0

        for x,target in zip(X,Y):

            out=1 if np.dot(x,w)+b>=0 else 0

            error=target-out

            w=w+lr*error*x
            b=b+lr*error

            total+=abs(error)

        if total==0:
            break

    print("Learning Rate:",lr)
    print("Epochs:",epoch+1)
    print("Weights:",w)
    print()
