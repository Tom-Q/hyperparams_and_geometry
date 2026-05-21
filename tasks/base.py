"""Abstract base class for all tasks."""
from abc import ABC, abstractmethod


class Task(ABC):
    name: str
    paradigm: str               # "supervised" | "rl" | "rnn"
    input_size: int             # flat dim for MLP/RL; per-step dim for RNN
    output_size: int
    n_steps: int | None         # RNN sequence length; None for MLP/RL
    hidden_size_range: tuple[int, int]
    success_threshold: float
    metric_name: str            # "val_acc" | "mean_return" | "val_mse"
    rdm_time_indices: list | None = None   # time steps to save for RNN tasks
    max_steps: int | None = None           # max env steps for RL tasks

    @abstractmethod
    def get_data(self, data_dir: str, seed: int = 42):
        """Return (train_loader, val_loader) for supervised/RNN,
        or an env-factory callable for RL."""

    @abstractmethod
    def get_rdm_stimuli(self, data_dir: str, seed: int = 42):
        """Return (inputs_array, metadata_dict).
        inputs shape: (N, input_size)         for MLP/RL
                      (N, n_steps, input_size) for RNN
        """

    @abstractmethod
    def categorical_space(self) -> dict[str, list]:
        """Ordered dict of categorical hyperparameter name → list of choices."""

    def make_loss(self):
        """Return the PyTorch loss function for this task (supervised/RNN only)."""
        raise NotImplementedError(f"{self.name} has no make_loss()")
