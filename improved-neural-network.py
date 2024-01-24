import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# start with data X, y and split, then split again to cross-validation and test
# 50% 40% 10%
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)

lambda = 0.1

model = Sequential(
    [
        Dense(120, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda)),
        Dense(40, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lambda)),
        Dense(classes, activation = 'linear')
    ]
)
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=1000
)

yhat_train = model.predict(X_train)
yhat_cv = model.predict(X_cv)

error_train = model.mse(y_train, yhat_train)
error_cv = model.mse(y_cv, yhat_cv)

# if the error_train is low compared to some benchmark, but the error_cv is high, the model has high variance
# if error_train and error_cv are high compared to some benchmark, the model has high bias

# Methods to fix high bias:
# 1. reduce the regularization parameter (lambda)
# 2. review the specific errors and find if there are certain types of common errors
    # 2. add polynomial feature(s) based on those errors
    # 3. add another feature based on those errors

# Methods to fix high variance:
# 1. increase the regularization parameter (lambda)
# 2. get more data (specifically on where the model makes wrong predictions)
# 3. reduce the number of features
