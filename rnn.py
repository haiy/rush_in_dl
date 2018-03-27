import numpy as np

sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec) / np.sum(np.exp(x_vec), axis=1, keepdims=True)

# 根据当前输入字母，预测下一个字母
words = "today is a good day, hello how are you fine thanks and you what's your name"

all_data = list(words)
data_size = len(all_data)
vocab = list(set(all_data))
vocab_size = len(vocab)

char_idx = {char: idx for char, idx in enumerate(vocab)}
idx_char = {idx: char for char, idx in enumerate(vocab)}

seq_len = 26
hidden_size = 100

curr_pos = 0
n = 0

U = np.random.rand(hidden_size, vocab_size)
W = np.random.rand(hidden_size, hidden_size)
V = np.random.rand(vocab_size, hidden_size)
s_t_pre = np.random.rand(hidden_size, 1)


while curr_pos + seq_len + 1 < data_size:
    # x row char vector: seq_len
    raw_x = all_data[curr_pos:curr_pos + seq_len]
    raw_y = all_data[curr_pos + 1:curr_pos + seq_len + 1]

    x_all,y_p_all,s_all = {},{},{}


    # forward calculate
    loss = 0
    # one hot column vector: vocab_size*1
    x_t = np.zeros([vocab_size, 1])
    y_t = np.zeros([vocab_size, 1])
    for t in range(seq_len):
        x_t[char_idx[x[t]]] = [1]
        y_t_idx = char_idx[y[t]]
        y_t[y_t_idx] = [1]

        # 这个地方的矩阵大小不好记住，可以尝试用倒推的方式
        # hd_sz*1 = (hd_sz*vb_sz dot  vb_sz*1) + (hd_sz*hd_sz dot hd_sz*1)
        s_t = tanh_f(np.dot(U, x_t) + np.dot(W, s_t_pre))
        # vocab_size*1 = (vb_sz*hd_z dot hd_sz*1)
        y_p_t = softmax_f(np.dot(V, s_t))
        loss += -np.log(y_p_t[y_t_idx])

        # do the recurrent stat assign
        s_t_pre = s_t

        # just store all the mid values
        x_all[t] = x_t
        y_p_all[t] = y_p_t
        s_all[t] = s_t

    print("loss -> ", loss)
    # backprogate calculate
    t = 0
    for t in reversed(range(seq_len)):
        pass



