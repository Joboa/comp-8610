from typing import List
import torch.nn as nn
from torch.functional import F


class MNISTANN(nn.Module):
    def __init__(self, input_size: int, hidden_layer_sizes: List[int], output_size: int):
        super().__init__()

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.model(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        # Dropout regularization
        def dropout_layer(p=0.3):
            return nn.Dropout2d(p=p) if use_dropout else nn.Identity()

        # Conv layers
        # image size (N, RGB(3), 32, 32)
        self.conv1 = nn.Conv2d(3, 6, 5)  # (input channel, output channel, kernel size)
        self.dropout1 = dropout_layer()
        self.pool = nn.MaxPool2d(2, 2)  # kernel size=2, stride = 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = dropout_layer()

        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output for the CIFAR10 classes

    def forward(self, x):
        x = self.droput1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2(F.relu(self.conv2(x)))

        # Moving to fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
