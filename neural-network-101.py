import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
# from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
# from tensorflow.keras.activations import sigmoid

# linear model

linear_layer = Dense(units=1, activation = 'linear', )
a1 = linear_layer(data) # a1 is the first layer's activations
w,b = linear_layer.get_weights()
linear_layer.set_weights([starting_w, starting_b])
prediction = linear_layer(x_train)

# logistic model
model = Sequential([
    Dense(1, input_dim=1, activation = 'sigmoid', name="L1")
])

# summarize model
model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()

logistic_layer.set_weights([starting_w, starting_b])
w,b = logistic_layer.get_weights()

a1 = model.predict(x_train)
