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
sample_size = len(input_data)

reglamda = 12
epsion = 0.0001


def train(input_x):
    W1 = np.random.rand(2, 5)
    b1 = np.zeros((1, 5))
    W2 = np.random.rand(5, 6)
    b2 = np.zeros((1, 6))
    W3 = np.random.rand(6, 2)
    b3 = np.zeros((1, 2))

    # input*4*6*2
    for i in range(0, 20000):
        # forward
        h1 = np.dot(input_x, W1) + b1  # h1 4*2 dot 2*5 = 4*5
        o1 = sigmoid_f(h1)
        # print("o1 shape ->", o1.shape, o1)
        h2 = np.dot(o1, W2) + b2  # h2 4*5 dot 5*6 = 4*6
        o2 = tanh_f(h2)
        # print("o2 shape ->", o2.shape, o2)
        h3 = np.dot(o2, W3) + b3  # h3 4*6 dot 6*2 = 4*2
        o3 = softmax_f(h3)
        # print("o3 shape ->", o3.shape, o3)

        # calculate loss
        loss = np.sum(-np.log(o3[range(sample_size), label]))
        loss += reglamda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2) + np.sum(np.square(W3))))

        # backward
        delta3 = o3 - label_info  # 4*2
        dw3 = (o2.T).dot(delta3)  # 6*2
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W3.T) * (1 - np.power(o2, 2))  # 4*2 6*2 => 4*6
        dw2 = (o1.T).dot(delta2)  # 4*5 4*6 => 5*6
        db2 = np.sum(delta2, axis=0, keepdims=True)
        # 4*6 5*6 4*5 =>4*5
        delta1 = delta2.dot(W2.T) * (sigmoid_f(h1) - np.power(sigmoid_f(h1), 2))
        dw1 = (input_x.T).dot(delta1)  # 4*2 4*5 => 2*5
        db1 = np.sum(delta1, axis=0, keepdims=True)  # 4*1

        # 对正则化项的偏导
        dw1 += reglamda * W1
        dw2 += reglamda * W2
        dw3 += reglamda * W3

        # 梯度更新部分
        W1 += -epsion * dw1
        b1 += -epsion * db1
        W2 += -epsion * dw2
        b2 += -epsion * db2
        W3 += -epsion * dw3
        b3 += -epsion * db3

        if (i % 50 == 0):
            print("loss -> ", loss)
    return  { "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3 }

model_weight = train(input_x)
print(model_weight)
