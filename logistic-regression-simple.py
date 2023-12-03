import numpy as np

# outputs a scalar between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb = sigmoid(z_i)
        cost += y[i] * np.log(f_wb) + (1 - y[i]) * np.log(1 - f_wb)

    cost /= -m
    return cost


x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost(x_train, y_train, w_tmp, b_tmp))
