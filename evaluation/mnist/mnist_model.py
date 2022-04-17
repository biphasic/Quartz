import torch.nn as nn


class ConvNet(nn.Sequential):
    def __init__(self):
        super().__init__(
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                # nn.Dropout2d(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(6, 12, kernel_size=5),
                nn.ReLU(),
                # nn.Dropout2d(0.4),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(12, 120, kernel_size=4),
                nn.ReLU(),
                # nn.Dropout2d(0.4),
                nn.Flatten(),
                nn.Linear(120, 10),
        )

