import numpy as np

sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec - np.max(x_vec)) / np.sum(np.exp(x_vec - np.max(x_vec)))

words = "today is a good day, hello how are you fine thanks and you what's your name"

all_data = list(words)
data_size = len(all_data)
vocab = list(set(all_data))
char_idx = {char: idx for idx, char in enumerate(vocab)}
idx_char = {idx: char for idx, char in enumerate(vocab)}
train_data = [char_idx[w] for w in all_data]

seq_len = 26  # t len
hidden_size = 1000
vocab_size = len(vocab)  # 21
Z = vocab_size + hidden_size

# hd_sz*(vocab_sz + hd_sz)

W_i = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / Z), (hidden_size, Z))
W_f = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / Z), (hidden_size, Z))
W_o = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / Z), (hidden_size, Z))
W_cc = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / Z), (hidden_size, Z))

W_hy = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / hidden_size), (vocab_size, hidden_size))

learning_rate = 0.00001
epoch = 0
while epoch < 6:
    pos = 0

    while pos + 1 < len(train_data):
        loss = 0
        temp_cache = {}
        # hd_sz * 1
        c_prev = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, 1))
        h_prev = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, 1))
        temp_cache[-1] = (0, 0, 0, 0, 0, c_prev, h_prev, 0, 0, 0)
        for t in range(seq_len):
            x_t = np.zeros([vocab_size, 1])
            x_t[train_data[pos]] = [1]
            y_idx = train_data[pos + 1]
            X = np.vstack([x_t, temp_cache[t - 1][6]])  # (vocab_sz+hidden_size)*1

            # hd_sz * 1
            f_t = sigmoid_f(np.dot(W_f, X))
            i_t = sigmoid_f(np.dot(W_i, X))
            o_t = sigmoid_f(np.dot(W_o, X))
            cc_t = tanh_f(np.dot(W_cc, X))

            # hd_sz*1
            c_t = np.multiply(f_t, temp_cache[t - 1][5]) + np.multiply(i_t, cc_t)
            h_t = np.multiply(o_t, tanh_f(c_t))

            y_t = np.dot(W_hy, h_t)  # vb_sz*hd_dz  hd_sz*1
            prob = softmax_f(y_t)
            loss += -np.sum(np.log(prob[y_idx])) / vocab_size

            temp_cache[t] = (X, f_t, i_t, o_t, cc_t, c_t, h_t, y_t, prob, y_idx)

        print("here %d loss %f->" % (pos, loss))
        pos += 1

        # backward
        d_w_f = np.zeros_like(W_f, dtype=np.float)
        d_w_i = np.zeros_like(W_i, dtype=np.float)
        d_w_o = np.zeros_like(W_o, dtype=np.float)
        d_w_cc = np.zeros_like(W_cc, dtype=np.float)

        prev_dct = np.zeros((hidden_size, 1), dtype=float)
        for t in reversed(range(seq_len)):
            X_t, f_t, i_t, o_t, cc_t, c_t, h_t, y_t, prob, y_idx = temp_cache[t]
            y = np.zeros([vocab_size, 1])
            y[y_idx] = [1]
            dy = prob - y  # vb_sz * 1
            d_w_hy = np.dot(dy, h_t.T)  # vb_sz*1 hd_sz*1  = vb_sz*hd_sz
            dh_t = np.dot(dy.T, W_hy)  # # 1*vb_sz vb_sz*hd_sz = 1 * hdsz
            do_t = np.multiply(dh_t, tanh_f(c_t))  # 1 * hd_sz  hd_sz*1 = hd_sz*hd_sz
            # hd_sz * hd_sz  hd_sz*1 = hd_sz * 1
            dc_t = np.dot(do_t, np.multiply(o_t, 1 - np.square(tanh_f(c_t)))) + prev_dct
            prev_dct = dc_t
            df_t = np.dot(temp_cache[t - 1][5], dc_t.T)  # hd_sz * 1 hd_sz*1 = hd_sz*hd_sz
            di_t = np.dot(cc_t, dc_t.T)
            d_cc_t = np.dot(i_t, dc_t.T)
            d_c_prev = np.dot(f_t, dc_t.T)
            # hd_sz * (hd_sz + vb_sz) = hd_sz_sz*hd_sz hd_sz*1   (vocab_sz+hidden_size)*1
            d_w_f += np.dot(df_t, f_t - np.square(f_t)).dot(X_t.T)
            d_w_i += np.dot(di_t, i_t - np.square(i_t)).dot(X_t.T)
            d_w_o += np.dot(do_t, o_t - np.square(o_t)).dot(X_t.T)
            d_w_cc += np.dot(d_cc_t, 1 - np.square(tanh_f(cc_t))).dot(X_t.T)

        W_f -= learning_rate * d_w_f
        W_i -= learning_rate * d_w_i
        W_o -= learning_rate * d_w_o
        W_cc -= learning_rate * d_w_cc
