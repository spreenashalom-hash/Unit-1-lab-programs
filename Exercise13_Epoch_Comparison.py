from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])

for ep in [10,30,50,100]:

    model=Sequential([
        Dense(4,input_dim=2,activation='relu'),
        Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history=model.fit(X,Y,epochs=ep,verbose=0)

    plt.plot(history.history['accuracy'],label=str(ep))

plt.legend()
plt.show()
