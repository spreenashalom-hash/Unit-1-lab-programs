import numpy as np
import matplotlib.pyplot as plt

weights = np.array([-1,-1])
bias = 0.5

def predict(x):
    return 1 if np.dot(x,weights)+bias>=0 else 0

inputs=np.array([[0,0],[0,1],[1,0],[1,1]])

for x in inputs:
    print(x,predict(x))

x=np.linspace(-1,2,100)
y=(-weights[0]*x-bias)/weights[1]

plt.scatter(inputs[:,0],inputs[:,1],c=['red','blue','blue','blue'],s=80)
plt.plot(x,y)
plt.grid()
plt.show()
