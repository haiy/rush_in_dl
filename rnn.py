import numpy as np

sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec-np.max(x_vec)) / np.sum(np.exp(x_vec-np.max(x_vec)))

# 根据当前输入字母，预测下一个字母
words = "today is a good day, hello how are you fine thanks and you what's your name"

all_data = list(words)
data_size = len(all_data)
vocab = list(set(all_data))
vocab_size = len(vocab)
char_idx = {char: idx for idx,char in enumerate(vocab)}
idx_char = {idx: char for idx,char in enumerate(vocab)}

seq_len = 26
hidden_size = 100
n = 0
learning_rate = 0.0001

U = np.random.rand(hidden_size, vocab_size)
W = np.random.rand(hidden_size, hidden_size)
V = np.random.rand(vocab_size, hidden_size)
s_t_pre = np.random.rand(hidden_size, 1)

U = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / vocab_size), (hidden_size, vocab_size))
W = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
V = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (vocab_size, hidden_size))

epoch = 0
while epoch < 6:
    curr_pos = 0
    curr_iter = 0
    while curr_pos + seq_len + 1 < data_size:
        loss = 0.00
        # x row char vector: seq_len
        raw_x = all_data[curr_pos:curr_pos + seq_len]
        raw_y = all_data[curr_pos + 1:curr_pos + seq_len + 1]
        x = [char_idx[c] for c in raw_x]
        y = [char_idx[c] for c in raw_y]

        y_p_all = np.matrix(np.zeros([seq_len, vocab_size]))
        s_all = np.matrix(np.zeros([seq_len+1, hidden_size]))
        s_all[-1] = s_t_pre.T

        # forward calculate
        # one hot column vector: vocab_size*1
        for t in range(seq_len):
            x_t = np.zeros([vocab_size, 1])
            y_t = np.zeros([vocab_size, 1], dtype=np.int)
            x_t[x[t]] = [1]
            y_t[y[t]] = [1]
            # 这个地方的矩阵大小不好记住，可以尝试用倒推的方式
            # hd_sz*1 = (hd_sz*vb_sz dot  vb_sz*1) + (hd_sz*hd_sz dot hd_sz*1)
            s_t = tanh_f(np.dot(U, x_t) + np.dot(W, s_t_pre))
            # vocab_size*1 = (vb_sz*hd_z dot hd_sz*1)
            y_p_t = softmax_f(np.dot(V, s_all[t-1].T))
            # just store all the mid values
            y_p_all[t] = y_p_t.T
            s_all[t] = s_t.T

        loss = -np.sum(np.log(y_p_all[np.arange(len((y))), y]))/vocab_size
        print("epoch : %d %d loss -> %f" % (epoch, curr_iter, loss))
        curr_iter += 1

        # backprogate calculate
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        dU = np.zeros_like(U)
        t = 0
        for t in reversed(range(seq_len)):
            x_t = np.zeros([vocab_size, 1], dtype=np.int)
            y_t = np.zeros([vocab_size, 1], dtype=np.int)
            y_t[y[t]] = [1]
            x_t[x[t]] = [1]
            s_t = np.asarray(s_all[t].T)
            s_t_1 = np.asarray(s_all[t-1].T)
            y_p_t = np.asarray(y_p_all[t].T)
            # 21*21
            delta_y_t = y_p_t - y_t
            # dv 21*100
            dV += delta_y_t*s_t.T
            # V 21*100 delta_y 21*1     s_t 100*1 = 100*1
            delta_t =  np.dot(V.T, delta_y_t)*(1-np.square(s_t))
            for i in range(t):
                s_i = np.asarray(s_all[i].T)
                s_i_1 = np.asarray(s_all[i-1].T)
                # w 100*100 = 100*1 100*1
                dW += np.dot(delta_t, s_i_1.T)
                # U 100*21 100*1 21*1
                dU += np.dot(delta_t,x_t.T)
                # 100*1 = 100*100 100*1
                delta_t = W.dot(delta_t)*(1-np.square(s_i_1))
        curr_pos += 1
        W -= learning_rate*dW
        U -= learning_rate*dU
        V -= learning_rate*dV
    epoch += 1


