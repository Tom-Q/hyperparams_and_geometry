import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class DualTaskMNIST(Dataset):
    def __init__(self, images, digits, task_bits, labels):
        self.images = images        # (N, 784) float32
        self.task_bits = task_bits  # (N,) float32, 0 or 1
        self.labels = labels        # (N,) float32, 0 or 1
        self.digits = digits        # (N,) int, original digit

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.cat([self.images[idx], self.task_bits[idx].unsqueeze(0)])  # (785,)
        return x, self.labels[idx]


def _make_labels(digits, task_bit):
    if task_bit == 0:
        return (digits % 2 == 0).float()  # even=1, odd=0
    else:
        return (digits < 5).float()        # small=1, large=0


def load_mnist_splits(data_dir="data", seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1).float() / 255.0),
    ])

    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_images = mnist_train.data.float().view(-1, 784) / 255.0
    train_digits = mnist_train.targets
    test_images = mnist_test.data.float().view(-1, 784) / 255.0
    test_digits = mnist_test.targets

    # Stratified split of train into train/val/gp_test (80/10/10)
    indices = np.arange(len(train_digits))
    idx_train, idx_tmp = train_test_split(
        indices, test_size=0.2, stratify=train_digits.numpy(), random_state=seed
    )
    idx_val, idx_gp_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=train_digits[idx_tmp].numpy(), random_state=seed
    )

    def _build_dataset(images, digits):
        images = torch.tensor(images) if not isinstance(images, torch.Tensor) else images
        digits = torch.tensor(digits) if not isinstance(digits, torch.Tensor) else digits
        task0 = torch.zeros(len(images))
        task1 = torch.ones(len(images))
        lab0 = _make_labels(digits, 0)
        lab1 = _make_labels(digits, 1)
        all_images = torch.cat([images, images], dim=0)
        all_digits = torch.cat([digits, digits], dim=0)
        all_tasks = torch.cat([task0, task1], dim=0)
        all_labels = torch.cat([lab0, lab1], dim=0)
        return DualTaskMNIST(all_images, all_digits, all_tasks, all_labels)

    ds_train = _build_dataset(train_images[idx_train], train_digits[idx_train])
    ds_val = _build_dataset(train_images[idx_val], train_digits[idx_val])
    ds_gp_test = _build_dataset(train_images[idx_gp_test], train_digits[idx_gp_test])
    ds_test = _build_dataset(test_images, test_digits)

    return ds_train, ds_val, ds_gp_test, ds_test


def make_loader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
