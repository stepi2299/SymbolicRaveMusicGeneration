import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple, Type


class Encoder(nn.Module, ABC):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO check input data to encoder (symbolic music)
        # TODO Possible handle in some cases from symbolic to tensor (Tokenizer?)
        pass


class DummyEncoder(Encoder):
    def __init__(self, latent_dim: int) -> None:
        super(DummyEncoder, self).__init__(latent_dim=latent_dim)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(data.shape[0], self.latent_dim), torch.zeros(
            data.shape[0], self.latent_dim
        )


class SimpleConvEncoder(Encoder):
    def __init__(
        self, input_shape: Tuple[int, int, int], latent_dim: int, hidden_dim: int = 128
    ) -> None:
        super(SimpleConvEncoder, self).__init__(latent_dim=latent_dim)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.fc_mu = nn.Linear(hidden_dim * input_shape[1] * input_shape[2], latent_dim)
        self.fc_log_variance = nn.Linear(
            hidden_dim * input_shape[1] * input_shape[2], latent_dim
        )
        self.relu = nn.ReLU()

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(data)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, self.hidden_dim * self.input_shape[1] * self.input_shape[2])
        mu = self.fc_mu(x)
        log_variance = self.fc_log_variance(x)
        return mu, log_variance


def encoder_mapper(encoder_name: str) -> Type[Encoder]:
    if encoder_name == "simple_conv":
        return SimpleConvEncoder
    elif encoder_name == "dummy":
        return DummyEncoder
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}, possible: ['simple_conv']")
