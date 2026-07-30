from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
Dense(8,input_dim=2,activation='relu'),
Dense(4,activation='relu'),
Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

model.summary()
