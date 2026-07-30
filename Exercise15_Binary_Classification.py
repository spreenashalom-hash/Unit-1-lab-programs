from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X,Y=make_classification(n_samples=300,
                        n_features=3,
                        n_classes=2,
                        n_redundant=0,
                        random_state=42)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

model=Sequential([
Dense(8,input_dim=3,activation='relu'),
Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=100,verbose=0)

loss,acc=model.evaluate(X_test,Y_test)

print("Accuracy:",acc)
