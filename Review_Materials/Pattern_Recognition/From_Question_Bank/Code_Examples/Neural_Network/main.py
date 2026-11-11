import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(o, t):
    return 0.5 * (o - t) ** 2

# Initial weights
w1 = np.array([0.2, 0.3, 0.4, 0.5])
w2 = np.array([0.6, 0.7])

# Input and target
x = np.array([2, 4])
t = 1

# Forward propagation
h1 = sigmoid(np.dot(x, w1[:2]))
print("h1",h1)
h2 = sigmoid(np.dot(x, w1[2:]))
print("h2",h2)
z = np.dot([h1, h2], w2)
o = sigmoid(z)
print("o",o)

# Backward propagation
delta_o = (o - t) * o * (1 - o)
delta_h1 = delta_o * w2[0] * h1 * (1 - h1)
delta_h2 = delta_o * w2[1] * h2 * (1 - h2)

# Update weights
w1 -= np.array([delta_h1 * x[0], delta_h1 * x[1], delta_h2 * x[0], delta_h2 * x[1]])
w2 -= delta_o * np.array([h1, h2])

print("Updated weights w1:", w1.round(5))
print("Updated weights w2:", w2.round(5))