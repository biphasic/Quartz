import torch.nn as nn


class TimeWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()
        self.module = module

    def forward(self, data):
        first_dim = data.shape[0]

        data = data.flatten(start_dim=0, end_dim=1)

        data = self.module.forward(data)

        return data.unflatten(0, (first_dim, -1))

    def __repr__(self):
        return "Time wrapped: " + self.module.__repr__()