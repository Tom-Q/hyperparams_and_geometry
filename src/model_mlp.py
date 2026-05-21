import torch.nn as nn

ACTIVATIONS = {
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth, activation, init_scale):
        super().__init__()
        act_name = activation.lower()
        act_cls  = ACTIVATIONS[act_name]

        # Cap depth at 2 when hidden_size is very small (H//4 would be < 2)
        effective_depth = min(depth, 2) if hidden_size < 8 else depth
        self.effective_depth = effective_depth

        # Layer sizes: input → H → H//2 → H//4 → output_size
        sizes = [input_size]
        h = hidden_size
        for _ in range(effective_depth):
            sizes.append(max(1, h))
            h = h // 2
        sizes.append(output_size)

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:          # hidden layers only
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)
        self._hidden_act_indices = []
        self._find_act_indices(effective_depth)
        self._init_weights(act_name, init_scale)

    def _find_act_indices(self, depth):
        for layer_num in range(depth):
            self._hidden_act_indices.append(layer_num * 2 + 1)

    def _init_weights(self, act_name, init_scale):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if act_name == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                m.weight.data.mul_(init_scale)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def get_layer_activations(self, x):
        """Return {layer_0, layer_1, ...} of post-activation hidden representations."""
        activations = {}
        out = x
        act_count = 0
        for i, layer in enumerate(self.net):
            out = layer(out)
            if i in self._hidden_act_indices:
                activations[f"layer_{act_count}"] = out.detach().cpu()
                act_count += 1
        return activations
