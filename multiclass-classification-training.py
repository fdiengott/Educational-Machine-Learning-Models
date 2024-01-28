"""
The goal of this file is to be able to train a model for multiclass classification. The archetypal example is fitting
a model to the MNIST dataset, which includes images of handwritten numbers, so that it can properly classify each digit.
I will assume each image is 20x20 pixels. I have not attempted to optimize for the correct number of layers or units
(neurons), but am using this as an educational opportunity.
"""

import numpy as np
import tensorflow as tf
from tf.keras import Sequential
from tf.keras.layers import Dense
from tf.keras.losses import SparseCategoricalCrossentropy

X_train, y_train, X_test, y_test = load_data() # not yet implemented

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear') # this ideally would be softmax, but it is less numerically accurate
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X,y,epochs=100)

logits = model(X)
f_x = tf.nn.softmax(logits)

# to predict test data
prediction_probabilities = model.predict(X_test)

digit_predictions = np.argmax(prediction_probabilities)

# the less accurate way:
"""
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X,y,epochs=100)

model.predict(X_test)
"""

# Now, if we wanted to run diagnostics to see if we should have a different architecture, we should also have a separate dev set set aside. I'd use that instead of the test data set above to run predictions.
# Then I'd use the final test data to show benchmark results
