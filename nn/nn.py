import numpy as np

sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec) / np.sum(np.exp(x_vec), axis=1, keepdims=True)




input_data = np.asarray([
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0],
])

# 4*2
input_x = input_data[..., :2]
label = input_data[..., 2]
label_info = np.vstack([label, 1 - label])

# input*4*6*2

# h1 4*2 dot 2*5 = 4*5
W1 = np.random.rand(2, 5)
b1 = np.random.rand(5)
sum_1 = np.dot(input_x, W1) + b1
l1 = sigmoid_f(sum_1)
print("l1 shape ->", l1.shape)
print(l1)

# h2 4*5 dot 5*6 = 4*6
W2 = np.random.rand(5, 6)
b2 = np.random.rand(6)
sum_2 = np.dot(l1, W2) + b2
l2 = tanh_f(sum_2)
print("l2 shape ->", l2.shape)
print(l2)

# h3 4*6 dot 6*2 = 4*2
W3 = np.random.rand(6, 2)
b3 = np.random.rand(2)
sum_3 = np.dot(l2, W3) + b3
l3 = softmax_f(sum_3)
print("l3 shape ->", l3.shape)
print(l3)
