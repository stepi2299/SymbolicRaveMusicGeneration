import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple


class Decoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        pass

class SimpleConvDecoder(Decoder):
    def __init__(self, latent_dim: int, input_shape: Tuple[int, int, int], output_channels: int, hidden_dim: int = 128) -> None:
        super(SimpleConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, self.hidden_dim * input_shape[0] * input_shape[1])
        self.conv1 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent_vector)
        x = x.view(-1, self.hidden_dim, self.input_shape[0], self.input_shape[1])
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

