"""Shared hyperparameter space definitions for all task paradigms."""

SUPERVISED_CATS = {
    "batch_size": [1, 8, 512],
    "depth":      [1, 2, 3],
    "activation": ["sigmoid", "tanh", "relu"],
    "optimizer":  ["sgd", "adam"],
    "init_scale": [0.01, 1.0],
}

RNN_CATS = {
    "batch_size":   [8, 64],
    "cell_type":    ["rnn", "gru", "lstm"],
    "n_rnn_layers": [1, 2],
    "optimizer":    ["sgd", "adam"],
    "init_scale":   [0.01, 1.0],
}

RL_CATS = {
    "depth":      [1, 2, 3],
    "activation": ["sigmoid", "tanh", "relu"],
    "optimizer":  ["sgd", "adam"],
    "init_scale": [0.01, 1.0],
    "gamma":      [0.9, 0.99],
}
