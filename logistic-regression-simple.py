import numpy as np
import copy, math

# outputs a scalar between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb = sigmoid(z_i)
        cost -= y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

    cost /= m
    return cost


def getRegularizedCost(x,y,w,b,lambda_=1):
    m,n = x.shape
    cost = 0.

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb = sigmoid(z_i)
        cost -= y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

    cost /= m

    # the above can just be:
    # cost = compute_cost(x,y,w,b)

    regularizedCost = 0.
    for j in range(n):
        regularizedCost += w[j]**2

    regularizedCost *= (lambda_ * (2 * m))

    return cost + regularizedCost


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    return (x - mu) / sigma


def compute_gradient(x, y, w, b):
    m,n = x.shape
    derivative_w = np.zeros(n)
    derivative_b = 0.

    for i in range(m):
        f_wb = sigmoid(np.dot(x[i],w) + b)
        error = f_wb - y[i]
        for j in range(n):
            derivative_w += error * x[i, j]
        derivative_b += error

    derivative_w /= m
    derivative_b /= m

    return derivative_w, derivative_b

# this just replaces compute_gradient in gradient descent
def compute_gradient_regularized(x,y,w,b,lambda_ = 1):
    m,n = x.shape

    derivative_w, derivative_b = compute_gradient(x,y,w,b)

    for j in range(n):
        derivative_w += (lambda_ / m * w[j])

    return derivative_w, derivative_b


def gradient_descent(x,y,w_init,b_init,alpha,num_iterations):
    w = copy.deepcopy(w_init)
    b = b_init

    cost_history = []

    for i in range(num_iterations):
        derivative_w, derivative_b = compute_gradient(x,y,w,b)
        w = w - alpha * derivative_w
        w = w - alpha * derivative_b

        if i % math.ceil(num_iterations / 10) == 0:
            cost = compute_cost(x,y,w,b)
            cost_history.append(cost)
            print(f'cost: {cost:.2e}')

    return w, b, cost_history

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
num_examples = x_train.shape[0]

w_temp = np.zeros_like(num_examples)
b_temp = 0

x_norm = normalize(x_train)

print(f"initial cost: {compute_cost(x_norm, y_train, w_temp, b_temp)}")

w_final, b_final = gradient_descent(x_norm, y_train, w_temp, b_temp, 0.1, 1000)

print(w_final, b_final)
