import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cell_type,
                 n_rnn_layers, init_scale):
        super().__init__()
        self.cell_type = cell_type.lower()

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU}[self.cell_type]
        rnn_kwargs = dict(hidden_size=hidden_size, num_layers=1, batch_first=True)
        if self.cell_type == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"  # PyTorch default; explicit for clarity

        self.layers = nn.ModuleList([
            rnn_cls(input_size=(input_size if i == 0 else hidden_size), **rnn_kwargs)
            for i in range(n_rnn_layers)
        ])
        self.head = nn.Linear(hidden_size, output_size)
        self._init_weights(init_scale)

    def _init_weights(self, init_scale):
        for layer in self.layers:
            for name, p in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(p.data)
                    p.data.mul_(init_scale)
                elif "bias" in name:
                    nn.init.zeros_(p.data)
        nn.init.xavier_normal_(self.head.weight)
        self.head.weight.data.mul_(init_scale)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """x: (B, T, input_size) → logits: (B, output_size)."""
        out = x
        for layer in self.layers:
            out, _ = layer(out)       # out: (B, T, H)
        return self.head(out[:, -1])  # use final time step of last layer

    def get_step_activations(self, x):
        """x: (B, T, input_size) → list of n_rnn_layers tensors, each (B, T, H), detached CPU.
        Caller is responsible for setting eval/train mode around this call.
        For GRU/RNN, out is the sequence of hidden states (no separate cell state).
        """
        activations = []
        out = x
        with torch.no_grad():
            for layer in self.layers:
                out, _ = layer(out)   # (B, T, H)
                activations.append(out.detach().cpu())
        return activations
