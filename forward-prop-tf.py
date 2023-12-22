import numpy as np
import tensorflow as tf
from tensorflow.keras.modles import Sequential
from tensorflow.keras.layers import Dense

# let's assume we have a load dataset function
# X_train.shape (1000, 400). 1000 examples each with 400 pieces of data
# y_train.shape (1000, 1)
X_train, y_train = load_dataset()

# the shape of our neural network will be:
# 3 layers: 25 units, 15 units, 1 unit

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

model.fit(X_train,y_train, epochs=20)

predictions = model.predict(X_test)

y_test = (predictions >= 0.5).astype(int)
