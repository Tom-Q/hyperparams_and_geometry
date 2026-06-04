"""Shared hyperparameter space definitions for all task paradigms."""

# Continuous ranges: (low, high) on a log scale, shared across all paradigms.
LEARNING_RATE = (1e-5, 1e-1)
L1_REG        = (1e-6, 1e-1)
L2_REG        = (1e-6, 1e-2)

SUPERVISED_HYPERPARAMS = {
    "hidden_size": [16, 256],
    "batch_size":  [1, 64],
    "depth":       [1, 2],
    "activation":  ["sigmoid", "tanh", "relu"],
    "optimizer":   ["sgd", "adam"],
    "init_scale":  [0.1, 1.0],
}

RNN_HYPERPARAMS = {
    "hidden_size":  [16, 256],
    "batch_size":   [1, 64],
    "cell_type":    ["rnn", "gru"],
    "n_rnn_layers": [1, 2],
    "optimizer":    ["sgd", "adam"],
    "init_scale":   [0.1, 1.0],
}

RL_HYPERPARAMS = {
    "hidden_size": [16, 256],
    "depth":       [1, 2],
    "activation":  ["sigmoid", "tanh", "relu"],
    "optimizer":   ["sgd", "adam"],
    "init_scale":  [0.1, 1.0],
}
