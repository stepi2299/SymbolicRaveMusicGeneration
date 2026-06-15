from __future__ import annotations

import logging as lg
from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers import BertConfig

from src.models.utils.clamp import CLaMPModel, PatchEncoder
from src.utils import (
    base_weights_path,
    download_file,
    remove_state_dict_keys,
    rename_state_dict_keys,
)

logger = lg.getLogger(__name__)


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

    def load_pretrained(self):
        logger.warning(
            "This discriminator %s does not support loading pretrained weights.",
            self.name,
        )


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

        self.patch_embedding = torch.nn.Linear(patch_size * vocab_size, projection_layer)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.dense1 = nn.Linear(projection_layer * (sequence_length - 1), first_layer)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(first_layer, second_layer)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        self.dense3 = nn.Linear(second_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, music_tensor: torch.Tensor) -> torch.Tensor:
        # Accepts:
        #   integer tokens  [batch, seq_len] or [batch, seq_len, patch_size]
        #   soft one-hot    [batch, seq_len, vocab_size]  (float, from Gumbel-Softmax)
        if music_tensor.is_floating_point():
            # Already in probability form → reshape to [batch, -1, patch_size * vocab_size]
            music_embedding = music_tensor.reshape(len(music_tensor), -1, self.patch_size * self.vocab_size)
        else:
            music_embedding = (
                torch.nn.functional.one_hot(music_tensor, num_classes=self.vocab_size)
                .reshape(len(music_tensor), -1, self.patch_size * self.vocab_size)
                .float()
            )
        music_embedding = self.patch_embedding(music_embedding.to(self.device))
        music_embedding = music_embedding.view(music_embedding.size(0), -1)
        music_embedding = self.dense1(music_embedding)
        music_embedding = self.leaky_relu_1(music_embedding)
        music_embedding = self.dense2(music_embedding)
        music_embedding = self.leaky_relu_2(music_embedding)
        music_embedding = self.dense3(music_embedding)
        music_embedding = self.sigmoid(music_embedding)
        return music_embedding


class Clamp2Discriminator(Discriminator):
    """Discriminator backed by the CLaMP-2 music encoder.

    The CLaMP-2 patch-encoder extracts a global music embedding which is then
    passed through a single linear head + sigmoid to produce a real/fake score,
    matching the interface expected by the GAN training loop.
    """

    def __init__(
        self,
        clamp2_vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        sequence_length: int,
        patch_size: int,
        latent_dim: int,
        vocab_size: int,
        weights_url: str,
    ) -> None:
        super().__init__(
            name="clamp2",
            sequence_length=sequence_length,
            patch_size=patch_size,
            vocab_size=vocab_size,
        )
        self.latent_dim = latent_dim
        self.weights_path = base_weights_path / "clamp2.pth"
        self.weights_url = weights_url

        patchilizer_config = BertConfig(
            vocab_size=clamp2_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=sequence_length,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
        )
        self.patch_encoder = PatchEncoder(
            config=patchilizer_config,
            patch_size=patch_size,
            projection_layer=hidden_size,
            vocab_size=vocab_size,
        )
        self.model = CLaMPModel(patch_encoder=self.patch_encoder, latent_dim=latent_dim)
        # Binary classification head: latent vector → real/fake score
        self.classifier = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    # Pretrained weights (shared CLaMP-2 checkpoint with the encoder)
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        if self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        else:
            self._download_checkpoint()
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)

        model_checkpoint = checkpoint["model"]
        model_checkpoint = remove_state_dict_keys(model_checkpoint, ["text_model", "text_proj"])
        model_checkpoint = rename_state_dict_keys(model_checkpoint, "music_model", "patch_encoder")

        try:
            self.model.load_state_dict(model_checkpoint)
            logger.info(f"Successfully loaded the CLaMP-2 checkpoint into {self.name} discriminator")
        except RuntimeError as e:
            logger.warning(
                "Could not load the checkpoint in strict mode. " f"Received error: {e}. Loading in non-strict way."
            )
            if (
                self.model.patch_encoder.patch_embedding.weight.size()
                != model_checkpoint["patch_encoder.patch_embedding.weight"].size()
            ):
                logger.warning("Patch embedding weight size mismatch – removing from checkpoint.")
                model_checkpoint.pop("patch_encoder.patch_embedding.weight")
                model_checkpoint.pop("patch_encoder.patch_embedding.bias")
            if (
                self.model.patch_encoder.base.embeddings.position_embeddings.weight.size()
                != model_checkpoint["patch_encoder.base.embeddings.position_embeddings.weight"].size()
            ):
                logger.warning(
                    "Position embeddings weight size mismatch – removing from checkpoint."
                    f" Expected sequence len in CLAMP2: 512, received: {self.sequence_length}"
                )
                model_checkpoint.pop("patch_encoder.base.embeddings.position_embeddings.weight")
            self.model.load_state_dict(model_checkpoint, strict=False)

    def _download_checkpoint(self) -> None:
        logger.info(f"Downloading CLaMP-2 weights from {self.weights_url} into {self.weights_path}")
        download_file(self.weights_url, self.weights_path, verbose=True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a real/fake probability in [0, 1] for each sample.

        Args:
            data: music token tensor — one of:
                  ``[batch, seq_len, patch_size]`` integer (3-D),
                  ``[batch, seq_len]`` integer (2-D), or
                  ``[batch, seq_len, vocab_size]`` float (Gumbel-Softmax one-hot).

        Returns:
            Tensor of shape ``[batch, 1]`` with values in (0, 1).
        """
        if data.is_floating_point():
            # Soft one-hot input: derive mask from argmax (non-pad tokens)
            masks = (data.argmax(dim=-1) != 0).long()
        elif data.dim() == 3:
            masks = (data != 0).any(dim=-1).long()
        else:
            masks = (data != 0).long()

        hidden_states = self.model.get_music_features(music_inputs=data, music_masks=masks)  # [batch, latent_dim]

        return self.sigmoid(self.classifier(hidden_states))  # [batch, 1]


def discriminator_mapper(discriminator_name: str) -> type[Discriminator]:
    if discriminator_name == "dense":
        return DenseDiscriminator
    elif discriminator_name == "clamp2":
        return Clamp2Discriminator
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
