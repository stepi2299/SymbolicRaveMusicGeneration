import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type, Dict

from transformers import BertConfig
from src.models.clamp import CLaMPDecoderModel, PatchDecoder


class Decoder(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        name: str,
        patch_count: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.vocab_size = vocab_size
        self.patch_count = patch_count
        self.patch_size = patch_size

    @abstractmethod
    def forward(
        self, latent_vector: torch.Tensor, recv_in_music_form: bool = False
    ) -> torch.Tensor:
        pass

    @staticmethod
    def transform_to_music_form(recv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Transform the received tokens to the music form.
        Args:
            recv_tokens: [batch_size, patch_count, patch_size, vocab_size]
        Returns:
            music_form: [batch_size, patch_count, patch_size]
        """

        return recv_tokens.argmax(dim=-1)


class DummyDecoder(Decoder):
    def __init__(self, latent_dim: int, vocab_size: int) -> None:
        super(DummyDecoder, self).__init__(
            latent_dim=latent_dim,
            name="dummy",
            vocab_size=vocab_size,
            patch_count=64,
            patch_size=64,
        )

    def forward(
        self, latent_vector: torch.Tensor, recv_in_music_form: bool = False
    ) -> torch.Tensor:
        return torch.zeros(latent_vector.shape[0], 3, 64, 64)


class Clamp2Decoder(Decoder):
    """
    A higher-level "Decoder" that wraps CLaMPDecoderModel, similarly to how
    ClampEncoder wraps CLaMPModel. It receives a 1D latent vector and
    returns a 2D (or 3D) sequence of patch logits (or tokens).
    """

    def __init__(
        self,
        clamp2_vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        patch_count: int,
        patch_size: int,
        latent_dim: int,
        vocab_size: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            name="clamp2",
            patch_count=patch_count,
            patch_size=patch_size,
        )

        # Build a BERT config to match the "decode" dimension
        decoder_config = BertConfig(
            vocab_size=clamp2_vocab_size,  # Typically same as the encoder’s
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=patch_count,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
        )

        # Create the PatchDecoder
        self.patch_decoder = PatchDecoder(
            config=decoder_config,
            patch_size=patch_size,
            patch_count=patch_count,
            projection_layer=hidden_size,
            vocab_size=self.vocab_size,
        )

        # Wrap in CLaMP2DecoderModel
        self.model = CLaMPDecoderModel(
            patch_decoder=self.patch_decoder, latent_dim=latent_dim
        )

    def forward(
        self, latent_vector: torch.Tensor, recv_in_music_form: bool = False
    ) -> torch.Tensor:
        """
        Args:
            latent_vector: [batch_size, latent_dim]

        Returns:
            patch_logits: [batch_size, patch_count, patch_size, vocab_size]
        """
        # Construct an attention mask if desired. For a full-sequence decode, use all 1’s:
        attention_mask = torch.ones(
            (latent_vector.size(0), self.patch_count),
            dtype=torch.long,
            device=latent_vector.device,
        )  # TODO think about removing this or use to teach the model to ignore padding

        # Forward through the model
        patch_logits = self.model(
            latent_vectors=latent_vector,
            patch_count=self.patch_count,
            attention_mask=attention_mask,
        )
        patch_logits = patch_logits.view(
            latent_vector.size(0),  # batch_size
            self.patch_count,
            self.patch_size,
            self.vocab_size,
        )
        if recv_in_music_form:
            patch_logits = self.transform_to_music_form(patch_logits)
        return patch_logits


class ClampDecoder(Decoder):
    """
    A higher-level "Decoder" that wraps CLaMPDecoderModel, similarly to how
    ClampEncoder wraps CLaMPModel. It receives a 1D latent vector and
    returns a 2D (or 3D) sequence of patch logits (or tokens).
    """

    def __init__(
        self,
        patchilizer_config: Dict,
        patch_count: int,
        patch_size: int,
        latent_dim: int,
        vocab_size: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            name="clamp",
            patch_count=patch_count,
            patch_size=patch_size,
        )

        patchilizer_config = BertConfig(**patchilizer_config)
        self.patch_decoder = PatchDecoder(
            config=patchilizer_config,
            patch_size=patch_size,
            patch_count=patch_count,
            projection_layer=patchilizer_config.hidden_size,
            vocab_size=self.vocab_size,
        )

        # Wrap in CLaMP2DecoderModel
        self.model = CLaMPDecoderModel(
            patch_decoder=self.patch_decoder, latent_dim=latent_dim
        )

    def forward(
        self, latent_vector: torch.Tensor, recv_in_music_form: bool = False
    ) -> torch.Tensor:
        """
        Args:
            latent_vector: [batch_size, latent_dim]
        Returns:
            patch_logits: [batch_size, patch_count, patch_size, vocab_size]
        """
        attention_mask = torch.ones(
            (latent_vector.size(0), self.patch_count),
            dtype=torch.long,
            device=latent_vector.device,
        )  # TODO think about removing this or use to teach the model to ignore padding

        # Forward through the model
        patch_logits = self.model(
            latent_vectors=latent_vector,
            patch_count=self.patch_count,
            attention_mask=attention_mask,
        )
        patch_logits = patch_logits.view(
            latent_vector.size(0),  # batch_size
            self.patch_count,
            self.patch_size,
            self.vocab_size,
        )
        if recv_in_music_form:
            patch_logits = self.transform_to_music_form(patch_logits)
        return patch_logits


def decoder_mapper(decoder_name: str) -> Type[Decoder]:
    if decoder_name == "clamp2":
        return Clamp2Decoder
    elif decoder_name == "clamp":
        return ClampDecoder
    elif decoder_name == "dummy":
        return DummyDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
