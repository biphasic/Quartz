import torch.nn as nn


class TimeWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        batch_first: bool = True
    ):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, data):
        batch_size = data.shape[0 if self.batch_first else 1]

        data = data.flatten(start_dim=0, end_dim=1)

        data = self.module.forward(data)

        return data.unflatten(0, (batch_size, -1))
