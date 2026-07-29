import torch.nn as nn

CONV_OUT = 8192  # 128 filters × 8 × 8 after two 2×2 MaxPool on 32×32 input


def _conv_block(in_ch, out_ch, use_batchnorm):
    """Conv → (BN) → ReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


def _make_conv_frontend(use_batchnorm):
    """
    VGG-style 4-conv frontend:
        [Conv(3→64) → (BN) → ReLU] × 2 → MaxPool   # 64×16×16
        [Conv(64→128) → (BN) → ReLU] × 2 → MaxPool  # 128×8×8
        Flatten → 8192
    """
    layers = (
        _conv_block(3,   64,  use_batchnorm) +
        _conv_block(64,  64,  use_batchnorm) +
        [nn.MaxPool2d(2)] +
        _conv_block(64,  128, use_batchnorm) +
        _conv_block(128, 128, use_batchnorm) +
        [nn.MaxPool2d(2), nn.Flatten()]
    )
    return nn.Sequential(*layers)


class CNNMLP(nn.Module):
    """
    Fixed 4-conv VGG-style frontend + uniform-width FC backend.

    Conv frontend (never varied):
        Conv(3→64) → (BN) → ReLU → Conv(64→64) → (BN) → ReLU → MaxPool(2×2)
        Conv(64→128) → (BN) → ReLU → Conv(128→128) → (BN) → ReLU → MaxPool(2×2)
        Flatten → 8192

    FC backend (controlled by n_fc_layers, hidden_size, activation):
        Linear(8192, H) → act
        [Linear(H, H) → act] × (n_fc_layers - 1)
        Linear(H, 10)

    get_layer_activations returns post-activation outputs of FC hidden layers
    as {"layer_0": ..., "layer_1": ...}, matching the MLP interface.
    """

    def __init__(self, output_size: int, hidden_size: int,
                 n_fc_layers: int, activation: str, use_batchnorm: bool = True):
        super().__init__()
        self._n_fc_layers  = n_fc_layers
        self._activation   = activation.lower()

        self.conv = _make_conv_frontend(use_batchnorm)

        act_cls = nn.ReLU if self._activation == "relu" else nn.Tanh
        fc_layers = []
        in_size   = CONV_OUT
        for _ in range(n_fc_layers):
            fc_layers.append(nn.Linear(in_size, hidden_size))
            fc_layers.append(act_cls())
            in_size = hidden_size
        self.fc_hidden = nn.Sequential(*fc_layers)
        self.output    = nn.Linear(hidden_size, output_size)

        self._n_per_layer = 2  # (Linear, activation) per FC hidden layer
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.fc_hidden:
            if isinstance(m, nn.Linear):
                if self._activation == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc_hidden(x)
        return self.output(x)

    def get_layer_activations(self, x):
        """Return post-activation FC hidden representations as {"layer_0": ..., ...}."""
        x = self.conv(x)
        acts = {}
        for i in range(self._n_fc_layers):
            linear = self.fc_hidden[i * self._n_per_layer]
            act    = self.fc_hidden[i * self._n_per_layer + 1]
            x = act(linear(x))
            acts[f"layer_{i}"] = x.detach().cpu()
        return acts
