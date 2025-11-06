from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type


class Discriminator(nn.Module, ABC):
    def __init__(
        self,
        name: str,
        sequence_length: int,
        patch_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class DummyDiscriminator(Discriminator):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name, sequence_length=64, patch_size=64, vocab_size=128)
        self.model = nn.Sequential(
            nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)


class DenseDiscriminator(Discriminator):
    def __init__(
        self,
        patch_size: int,
        sequence_length: int,
        vocab_size: int,
        projection_layer: int,
        first_layer: int,
        second_layer: int,
    ) -> None:
        super().__init__(
            name="dense",
            sequence_length=sequence_length,
            patch_size=patch_size,
            vocab_size=vocab_size,
        )
        self.projection_layer = projection_layer

        self.patch_embedding = torch.nn.Linear(
            patch_size * vocab_size, projection_layer
        )
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.dense1 = nn.Linear(projection_layer * sequence_length, first_layer)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(first_layer, second_layer)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        self.dense3 = nn.Linear(second_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, music_tensor: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, patch_size]
        # Convert to embeddings
        music_embedding = torch.nn.functional.one_hot(
            music_tensor, num_classes=self.vocab_size
        )  # [batch_size, sequence_length, patch_size, vocab_size]
        music_embedding = music_embedding.reshape(
            len(music_embedding), -1, self.patch_size * self.vocab_size
        ).type(torch.FloatTensor)
        music_embedding = self.patch_embedding(music_embedding.to(self.device))
        music_embedding = music_embedding.view(music_embedding.size(0), -1)
        music_embedding = self.dense1(music_embedding)
        music_embedding = self.leaky_relu_1(music_embedding)
        music_embedding = self.dense2(music_embedding)
        music_embedding = self.leaky_relu_2(music_embedding)
        music_embedding = self.dense3(music_embedding)
        music_embedding = self.sigmoid(music_embedding)
        return music_embedding


class SimpleConvDiscriminator(Discriminator):
    def __init__(
        self,
        patch_size: int,
        sequence_length: int,
        vocab_size: int,
        projection_layer: int,
        kernel_size: int,
        stride: int,
        padding: int,
        linear_layer: int,
        out_conv_size: int,
        inner_conv_size: int,
    ) -> None:
        super().__init__(
            name="simple_conv",
            sequence_length=sequence_length,
            patch_size=patch_size,
            vocab_size=vocab_size,
        )
        self.projection_layer = projection_layer

        self.patch_embedding = torch.nn.Linear(
            patch_size * vocab_size, projection_layer
        )
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)

        self.conv1 = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=inner_conv_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=inner_conv_size,
            out_channels=out_conv_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Calculate output length after each conv layer
        def conv_output_length(input_length, k, s, p):
            return (input_length + 2 * p - k) // s + 1

        # Compute sequence lengths
        conv1_output_length = conv_output_length(
            projection_layer, kernel_size, stride, padding
        )
        conv2_output_length = conv_output_length(
            conv1_output_length, kernel_size, stride, padding
        )

        # Flattened features = channels * final_sequence_length
        self.flattened_features = out_conv_size * conv2_output_length

        # Dense layer
        self.dense = nn.Linear(self.flattened_features, linear_layer)

        # Output layer with single neuron
        self.output = nn.Linear(linear_layer, 1)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, music_tensor: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, patch_size]
        # Convert to embeddings
        music_embedding = torch.nn.functional.one_hot(
            music_tensor, num_classes=self.vocab_size
        )  # [batch_size, sequence_length, patch_size, vocab_size]
        music_embedding = music_embedding.reshape(
            len(music_embedding), -1, self.patch_size * self.vocab_size
        ).type(
            torch.FloatTensor
        )  # [batch_size, sequence_length, patch_size * vocab_size]
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
    elif discriminator_name == "dense":
        return DenseDiscriminator
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
