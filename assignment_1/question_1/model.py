import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self, act_type="linear"):
        super().__init__()

        self.activation = act_type

        if self.activation == "linear":
            self.activation = nn.Identity
        if self.activation == "sigmoid":
            self.activation = nn.Sigmoid
        if self.activation == "tanh":
            self.activation = nn.Tanh  

        self.linear = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            self.activation()
        )

    def forward(self, x):
        return self.linear(x)