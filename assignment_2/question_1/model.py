from typing import List
import torch.nn as nn

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