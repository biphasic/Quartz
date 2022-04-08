import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class ConvNet(nn.Sequential):
    def __init__(self):
        super().__init__(
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(6, 12, kernel_size=5),
                nn.ReLU(),
                nn.Dropout2d(0.4),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(12, 120, kernel_size=4),
                nn.ReLU(),
                nn.Dropout2d(0.4),
                nn.Flatten(),
                nn.Linear(120, 10),
        )

