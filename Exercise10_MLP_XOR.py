from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])

model=Sequential([
Dense(2,input_dim=2,activation='relu'),
Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

model.fit(X,Y,epochs=500,verbose=0)

print(model.predict(X))
