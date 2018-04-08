import numpy as np


sigmoid_f = lambda x_vec: 1 / (1 + np.exp(-x_vec))
tanh_f = lambda x_vec: 2 * sigmoid_f(2 * x_vec) - 1
softmax_f = lambda x_vec: np.exp(x_vec - np.max(x_vec)) / np.sum(np.exp(x_vec - np.max(x_vec)))


words = "today is a good day, hello how are you fine thanks and you what's your name"

all_data = list(words)
data_size = len(all_data)
vocab = list(set(all_data))
vocab_size = len(vocab)
char_idx = {char: idx for idx, char in enumerate(vocab)}
idx_char = {idx: char for idx, char in enumerate(vocab)}

seq_len = 26
hidden_size = 400
n = 0
learning_rate = 0.0001

rand_init = np.random.random
W_xi = rand_init([vocab_size, hidden_size])
W_hi = rand_init([hidden_size, hidden_size])

W_xf = rand_init([vocab_size, hidden_size])
W_hf = rand_init([hidden_size, hidden_size])

W_xo = rand_init([vocab_size, hidden_size])
W_ho = rand_init([hidden_size, hidden_size])

W_xc = rand_init([vocab_size, hidden_size])
W_hc = rand_init([hidden_size, hidden_size])




