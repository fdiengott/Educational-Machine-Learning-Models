import numpy as np

def dense(a_in, W, b, g):
    units = W.shape[0]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:,j]
        z = np.dot(a_in, w) + b[j]
        a_out[j] = g(z)

    return a_out

def dense_vectorized(a_in, W, b, g):
    z = np.matmul(a_in,W) + b
    a_out = g(z)

    return a_out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sequential(x, dense_fn, W1, b1, W2, b2, W3, b3):
    a1 = dense_fn(x, W1, b1, sigmoid)
    a2 = dense_fn(a1, W2, b2, sigmoid)
    a3 = dense_fn(a2, W3, b3, sigmoid)

    return a3

# let the below be pre-trained weights and biases
W1, b1, W2, b2, W3, b3 = get_weights()

# assuming we have X_test data
predictions = sequential(X_test, dense_vectorized, W1, b1, W2, b2, W3, b3)

predictions_refined = (predictions >= 0.5).astype(int)
