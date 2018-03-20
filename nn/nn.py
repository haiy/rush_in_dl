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
label_info = np.vstack([label, 1 - label]).T
sample_size=len(input_data)

W1 = np.random.rand(2, 5)
b1 = np.random.rand(5)


W2 = np.random.rand(5, 6)
b2 = np.random.rand(6)


W3 = np.random.rand(6, 2)
b3 = np.random.rand(2)

reglamda = 0.4

def train(input_x):
    # input*4*6*2
    # h1 4*2 dot 2*5 = 4*5
    # h2 4*5 dot 5*6 = 4*6
    # h3 4*6 dot 6*2 = 4*2

    # forward
    h1 = np.dot(input_x, W1) + b1
    o1 = sigmoid_f(h1)
    print("o1 shape ->", o1.shape)
    print(o1)
    h2 = np.dot(o1, W2) + b2
    o2 = tanh_f(h2)
    print("o2 shape ->", o2.shape)
    print(o2)
    h3 = np.dot(o2, W3) + b3
    o3 = softmax_f(h3)
    print("o3 shape ->", o3.shape)
    print(o3)

    loss = np.sum(-np.log(o3[range(sample_size),label]))
    print("loss -> ", loss)
    loss += reglamda/2*(np.sum(np.square(W1)) + np.sum(np.square(W2)  + np.sum(np.square(W3))))
    print("loss -> ", loss)

    #backward
    #4*2
    delta3 = o3 - label_info

    dw3 = (o2.T).dot(delta3)
    print("dw3 ->", dw3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    print("db3 ->", db3)
    delta2 = delta3.dot(W3.T)*(1-np.power(o2, 2))
    print("delta2 ->", delta2)
    # dw2 =

train(input_x)








