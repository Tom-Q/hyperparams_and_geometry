"""Shared hyperparameter space definitions for all task paradigms."""

SUPERVISED_CATS = {
    "hidden_size": [16, 256],
    "batch_size":  [1, 8, 64],
    "depth":       [1, 2],
    "activation":  ["sigmoid", "tanh", "relu"],
    "optimizer":   ["sgd", "adam"],
    "init_scale":  [0.1, 1.0],
}

RNN_CATS = {
    "hidden_size":  [16, 256],
    "batch_size":   [1, 64],
    "cell_type":    ["rnn", "gru"],
    "n_rnn_layers": [1, 2],
    "optimizer":    ["sgd", "adam"],
    "init_scale":   [0.1, 1.0],
}

RL_CATS = {
    "hidden_size": [16, 256],
    "depth":       [1, 2],
    "activation":  ["sigmoid", "tanh", "relu"],
    "optimizer":   ["sgd", "adam"],
    "init_scale":  [0.1, 1.0],
}
