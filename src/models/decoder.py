import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type

from transformers import BertConfig
from src.models.clamp2 import CLaMP2DecoderModel, PatchDecoder


class Decoder(nn.Module, ABC):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        pass


class DummyDecoder(Decoder):
    def __init__(self, latent_dim: int) -> None:
        super(DummyDecoder, self).__init__(latent_dim=latent_dim)

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latent_vector.shape[0], 3, 64, 64)


# class SimpleConvDecoder(Decoder):
#     def __init__(
#         self,
#         latent_dim: int,
#         input_shape: Tuple[int, int, int],
#         output_channels: int,
#         hidden_dim: int = 128,
#     ) -> None:
#         super(SimpleConvDecoder, self).__init__(latent_dim=latent_dim)
#         self.latent_dim = latent_dim
#         self.input_shape = input_shape
#         self.output_channels = output_channels
#         self.hidden_dim = hidden_dim
#
#         self.fc = nn.Linear(
#             latent_dim, self.hidden_dim * input_shape[0] * input_shape[1]
#         )
#         self.conv1 = nn.Conv2d(
#             in_channels=self.hidden_dim,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
#         )
#         self.conv3 = nn.Conv2d(
#             in_channels=32,
#             out_channels=output_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
#         x = self.fc(latent_vector)
#         x = x.view(-1, self.hidden_dim, self.input_shape[0], self.input_shape[1])
#         x = self.relu(x)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.sigmoid(x)
#         return x


class Clamp2Decoder(Decoder):
    """
    A higher-level "Decoder" that wraps CLaMP2DecoderModel, similarly to how
    Clamp2Encoder wraps CLaMP2Model. It receives a 1D latent vector and
    returns a 2D (or 3D) sequence of patch logits (or tokens).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        patch_length: int,
        patch_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.patch_length = patch_length
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # Build a BERT config to match the "decode" dimension
        decoder_config = BertConfig(
            vocab_size=vocab_size,  # Typically same as the encoder’s
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=patch_length,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
        )

        # Create the PatchDecoder
        self.patch_decoder = PatchDecoder(
            config=decoder_config, patch_size=patch_size, projection_layer=hidden_size
        )

        # Wrap in CLaMP2DecoderModel
        self.model = CLaMP2DecoderModel(
            patch_decoder=self.patch_decoder, latent_dim=latent_dim
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: [batch_size, latent_dim]
        Returns:
            patch_logits: [batch_size, patch_length, patch_size*128]
                (2D if you flatten the patch dimension, or 3D/4D if you reshape further)
        """
        # Construct an attention mask if desired. For a full-sequence decode, use all 1’s:
        attention_mask = torch.ones(
            (latent_vector.size(0), self.patch_length),
            dtype=torch.long,
            device=latent_vector.device,
        )

        # Forward through the model
        patch_logits = self.model(
            latent_vectors=latent_vector,
            seq_length=self.patch_length,
            attention_mask=attention_mask,
        )
        # patch_logits shape: [batch_size, patch_length, patch_size*128]

        # Optionally reshape to: [batch_size, patch_length, patch_size, 128]
        # patch_logits = patch_logits.view(
        #     patch_logits.size(0), patch_logits.size(1), self.patch_size, 128
        # )

        return patch_logits


def decoder_mapper(decoder_name: str) -> Type[Decoder]:
    if decoder_name == "clamp2":
        return Clamp2Decoder
    # elif decoder_name == "simple_conv":
    #     return SimpleConvDecoder
    elif decoder_name == "dummy":
        return DummyDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
