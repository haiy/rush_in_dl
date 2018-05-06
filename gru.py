import numpy as np

words = "hello how are you fine thanks and you what's your name.I'm lily.Hanmeimei.Good."
vocab = list(set(words))
vob_sz = len(vocab)
char_idx = dict(zip(vocab, list(range(vob_sz))))
train_data = [char_idx[c] for c in words]
hidden_sz = 100
seq_len = 10

W_zx = np.random.random((hidden_sz, vob_sz))
W_zh = np.random.random((hidden_sz, hidden_sz))
W_rx = np.random.random((hidden_sz, vob_sz))
W_rh = np.random.random((hidden_sz, hidden_sz))
W_cx = np.random.random((hidden_sz, vob_sz))
W_ch = np.random.random((hidden_sz, hidden_sz))
W_y = np.random.random((vob_sz, hidden_sz))

def onehot_x(x_val):
    x = np.zeros((vob_sz, 1))
    x[x_val] = [1]
    return x

sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec - np.max(x_vec)) / np.sum(np.exp(x_vec - np.max(x_vec)))

learning_rate = 0.001
pos = 0
while pos < len(train_data) - seq_len - 1:
    tmp_mem = {}
    L = 0
    # forward
    h_t_prev = np.zeros((hidden_sz,1))
    for t in range(seq_len):
        cur_pos = t + pos
        x = onehot_x(train_data[cur_pos])
        y = train_data[cur_pos + 1]
        z_t = sigmoid_f(np.dot(W_zx, x) + np.dot(W_zh, h_t_prev))
        r_t = sigmoid_f(np.dot(W_rx, x) + np.dot(W_rh, h_t_prev))
        h_tc = tanh_f(np.dot(W_cx, x) + np.dot(np.multiply(W_ch, r_t), h_t_prev))
        h_t = np.multiply((1 - z_t), h_t_prev) + np.multiply(z_t, h_tc)
        y_t = softmax_f(np.dot(W_y, h_t))
        L_t = -np.log(y_t[y]) / vob_sz
        L += L_t
        tmp_mem[t] = (h_t_prev, z_t, r_t, h_tc, h_t, x, y)
        h_t_prev = h_t
    print("curr lost -> ", L)
    # init the boarder condition
    tmp_mem[-1] = map(lambda x: np.zeros_like(x), [h_t_prev, z_t, r_t, h_tc, h_t, x, y])
    tmp_mem[seq_len] = map(lambda x: np.zeros_like(x), [h_t_prev, z_t, r_t, h_tc, h_t, x, y])

    # backward
    d_Wzx = np.zeros_like(W_zx)
    d_Wzh = np.zeros_like(W_zh)
    d_Wrx = np.zeros_like(W_rx)
    d_Wrh = np.zeros_like(W_rh)
    d_Wcx = np.zeros_like(W_cx)
    d_Wch = np.zeros_like(W_ch)
    d_Wy = np.zeros_like(W_y)
    d_sig = lambda x: x - np.square(x)
    d_tanh = lambda x: 1 - np.square(x)
    d_sum_prev = 0  # \sum{dL_i/dh_t}
    for i in reversed(range(seq_len)):
        h_t_prev_p, z_t_p, r_t_p, h_tc_p, h_t_p, x_p, y_p = tmp_mem[i + 1]
        h_t_prev, z_t, r_t, h_tc, h_t, x_t, y_t = tmp_mem[i]
        h_t_prev_n, z_t_n, r_t_n, h_tc_n, h_t_n, x_n, y_n = tmp_mem[i - 1]

        # delta = dh_(t+1)/dh_t
        d_ht = np.dot(W_y.T, 1 - onehot_x(y_t))
        d_sig_z = d_sig(z_t)
        d_tanh_hp = d_tanh(h_tc_p)
        d_delta = np.dot(W_zh, np.multiply(h_tc_p - h_t, d_sig_z)) + \
                  np.multiply(np.multiply(z_t_p, d_tanh_hp),
                              np.dot(W_rh, np.multiply(np.dot(W_ch, h_t), d_sig_z)) + np.dot(W_rh, r_t)) + \
                  1 - z_t_p
        # d_sum(dL_i/dh_t)
        d_sum = d_sum_prev * d_delta + d_ht

        d_h_z = np.multiply((h_tc - h_t_n), d_sig_z)
        d_Wzx += np.dot(np.multiply(d_sum, d_h_z), x_t.T)
        d_Wzh += np.dot(np.multiply(d_sum, d_h_z), h_t_n.T)

        d_h_r = np.multiply(z_t, np.multiply(d_tanh(h_tc), np.dot(W_ch, h_t_n)))
        d_h_r = np.multiply(d_h_r, d_sig(r_t))
        d_Wrx += np.dot(np.multiply(d_sum, d_h_r), x_t.T)
        d_Wrh += np.dot(np.multiply(d_sum, d_h_r), h_t_n.T)

        d_h_c = np.multiply(z_t, d_tanh(h_tc))
        d_Wcx += np.dot(np.multiply(d_sum, d_h_c), x_t.T)
        d_Wch += np.dot(np.multiply(d_sum, np.multiply(d_h_c, np.multiply(r_t, h_t_n))), h_t_n.T)
        d_Wy += np.dot(1 - onehot_x(y_t), h_t.T)
    W_zx -= learning_rate * d_Wzx
    W_zh -= learning_rate * d_Wzh
    W_rx -= learning_rate * d_Wrx
    W_rh -= learning_rate * d_Wrh
    W_cx -= learning_rate * d_Wcx
    W_ch -= learning_rate * d_Wch
    W_y -= learning_rate * d_Wy
    pos += 1
