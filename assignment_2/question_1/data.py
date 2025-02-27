from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def mnist_data(batch_size):
    """Returns MNIST datasets and dataloader"""

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        # (mean, std) => (img - mean) / std == [-1,1]
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data/mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def cifar_data(batch_size):
    """Returns CIFAR datasets and dataloader"""

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        # (mean, std) => (img - mean) / std == [-1,1]
        transforms.Normalize((0.5,), (0.5,)),
        # transforms.Lambda(lambda x: (x + 1) / 2) # [-1, 1] to [0, 1] [for visualization]
    ])

    train_dataset = datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader