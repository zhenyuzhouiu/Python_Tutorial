# using numpy
# implement a simple linear regression prediction
# firstly implement by manually

import numpy as np

x = np.array([1, 2, 3, 4], dtype=np.float32)

y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0

# forward
def forward(x, w):

    y_hat = x * w
    return y_hat

# mse loss
def loss(y_hat, y):
    z = (y_hat - y)**2
    return z.mean()

# gradient
# MSE = 1/N * ( w*x -y) ^ 2
# dJ/dw = 2/N*x (w*x - y)
def gradient(x, y, y_hat):
    grad = np.dot(2*x, y_hat-y)
    return grad.mean()

l_rate = 0.01
epoch = 10

print(f'Prediction before training: f(5) = {forward(5, w):.3f}')

for i in range(epoch):
    y_hat = forward(x, w)

    l = loss(y_hat, y)

    dw = gradient(x, y, y_hat)

    w = w- l_rate*dw

    if epoch % 1 == 0:
        print(f'epoch {i+1}: w = {w:.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {forward(5, w):.3f}')

