from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type


class Discriminator(nn.Module, ABC):
    def __init__(
        self,
        name: str,
        patch_len: int,
        patch_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.patch_size = patch_size
        self.patch_len = patch_len
        self.vocab_size = vocab_size

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class DummyDiscriminator(Discriminator):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name, patch_len=64, patch_size=64, vocab_size=128)
        self.model = nn.Sequential(
            nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)


class SimpleConvDiscriminator(Discriminator):
    def __init__(
        self, patch_size: int, patch_len: int, vocab_size: int, projection_layer: int
    ) -> None:
        super().__init__(
            name="simple_conv",
            patch_len=patch_len,
            patch_size=patch_size,
            vocab_size=vocab_size,
        )
        self.projection_layer = projection_layer
        self.vocab_size = vocab_size

        self.patch_embedding = torch.nn.Linear(
            patch_size * vocab_size, projection_layer
        )
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)

        self.conv1 = nn.Conv1d(
            in_channels=patch_len,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Calculate the number of features after flattening
        self.flattened_features = 128 * (projection_layer // 4)

        # Dense layer
        self.dense = nn.Linear(self.flattened_features, 256)

        # Output layer with single neuron
        self.output = nn.Linear(256, 1)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, music_tensor: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, patch_length, patch_size]
        # Convert to embeddings
        music_embedding = torch.nn.functional.one_hot(
            music_tensor, num_classes=self.vocab_size
        )  # [batch_size, patch_len, patch_size, token_size]
        music_embedding = music_embedding.reshape(
            len(music_embedding), -1, self.patch_size * self.vocab_size
        ).type(torch.FloatTensor)  # [batch_size, patch_len, patch_size * token_size]
        music_embedding = self.patch_embedding(music_embedding.to(self.device))

        input_processing_tensor = self.conv1(music_embedding)
        input_processing_tensor = self.leaky_relu(input_processing_tensor)

        input_processing_tensor = self.conv2(input_processing_tensor)
        input_processing_tensor = self.leaky_relu(input_processing_tensor)

        # Flatten the output
        input_processing_tensor = input_processing_tensor.view(
            input_processing_tensor.size(0), -1
        )
        input_processing_tensor = self.dense(input_processing_tensor)
        input_processing_tensor = self.leaky_relu(input_processing_tensor)

        input_processing_tensor = self.output(input_processing_tensor)
        recv_classification_value = self.sigmoid(input_processing_tensor)

        return recv_classification_value


def discriminator_mapper(discriminator_name: str) -> Type[Discriminator]:
    if discriminator_name == "dummy":
        return DummyDiscriminator
    elif discriminator_name == "simple_conv":
        return SimpleConvDiscriminator
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
