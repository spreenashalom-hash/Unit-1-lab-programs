import numpy as np

gates={
'AND':([1,1],-1.5),
'OR':([1,1],-0.5),
'NAND':([-1,-1],1.5)
}

inputs=np.array([[0,0],[0,1],[1,0],[1,1]])

for gate,(w,b) in gates.items():
    print("\n",gate)
    for x in inputs:
        y=1 if np.dot(x,w)+b>=0 else 0
        print(x,y)
