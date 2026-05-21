from .base import Task
from .mnist_dual import MNISTDualTask
from .mnist_10way import MNIST10WayTask
from .fashion_10way import Fashion10WayTask
from .spirals import SpiralsTask
from .parity import ParityTask
from .cartpole import CartPoleTask
from .fourrooms import FourRoomsTask
from .mnist_rnn import MNISTRNNTask
from .adding import AddingTask

TASKS: dict[str, type] = {
    "mnist_dual":    MNISTDualTask,
    "mnist_10way":   MNIST10WayTask,
    "fashion_10way": Fashion10WayTask,
    "spirals":       SpiralsTask,
    "parity":        ParityTask,
    "cartpole":      CartPoleTask,
    "fourrooms":     FourRoomsTask,
    "mnist_rnn":     MNISTRNNTask,
    "adding":        AddingTask,
}

__all__ = ["TASKS", "Task",
           "MNISTDualTask", "MNIST10WayTask", "Fashion10WayTask",
           "SpiralsTask", "ParityTask",
           "CartPoleTask", "FourRoomsTask",
           "MNISTRNNTask", "AddingTask"]
