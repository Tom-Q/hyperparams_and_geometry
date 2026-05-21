import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cell_type,
                 n_rnn_layers, init_scale):
        super().__init__()
        self.cell_type = cell_type.lower()

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[self.cell_type]
        rnn_kwargs = dict(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = n_rnn_layers,
            batch_first = True,
        )
        if self.cell_type == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"
        self.rnn = rnn_cls(**rnn_kwargs)
        self.head = nn.Linear(hidden_size, output_size)
        self._init_weights(init_scale)

    def _init_weights(self, init_scale):
        for name, p in self.rnn.named_parameters():
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
        out, _ = self.rnn(x)          # out: (B, T, H)
        return self.head(out[:, -1])  # use final time step

    def get_step_activations(self, x):
        """x: (B, T, input_size) → list of T tensors, each (B, H), detached CPU.
        Caller is responsible for setting eval/train mode around this call.
        For LSTM, out is the sequence of hidden states (not cell states).
        """
        with torch.no_grad():
            out, _ = self.rnn(x)      # (B, T, H)
        return [out[:, t, :].detach().cpu() for t in range(out.shape[1])]
