from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple, Type, Dict
import logging as lg
import requests
from tqdm import tqdm

from transformers import BertConfig
from src.models.clamp import PatchEncoder, CLaMPModel
from src.utils import (
    download_file,
    get_possible_device,
    base_weights_path,
    remove_state_dict_keys,
    rename_state_dict_keys,
)

logger = lg.getLogger(__name__)


class Encoder(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        sequence_length: int,
        patch_size: int,
        name: str,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size

    @abstractmethod
    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def load_pretrained(self) -> None:
        logger.info(f"There is not pretrained model for this: {self.name} encoder")


class DummyEncoder(Encoder):
    def __init__(self, latent_dim: int, vocab_size: int) -> None:
        super(DummyEncoder, self).__init__(
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            name="dummy",
            patch_size=64,
            sequence_length=64,
        )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(data), self.latent_dim), torch.zeros(
            data.shape[0], self.latent_dim
        )


class Clamp2Encoder(Encoder):
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
        super(Clamp2Encoder, self).__init__(
            latent_dim=latent_dim,
            name="clamp2",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
        )
        self.device = get_possible_device()
        self.weights_path = base_weights_path / "clamp2.pth"
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
            vocab_size=self.vocab_size,
        )
        self.model = CLaMPModel(patch_encoder=self.patch_encoder, latent_dim=latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.weights_url = weights_url

    def load_pretrained(self) -> None:
        if self.weights_path.exists():
            checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        else:
            self._download_checkpoint()
            checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        model_checkpoint = checkpoint["model"]
        model_checkpoint = remove_state_dict_keys(
            model_checkpoint, ["text_model", "text_proj"]
        )
        model_checkpoint = rename_state_dict_keys(
            model_checkpoint, "music_model", "patch_encoder"
        )
        try:
            self.model.load_state_dict(model_checkpoint)
            logger.info(f"Successfully loaded the checkpoint to the model: {self.name}")
        except RuntimeError as e:
            logger.warning(
                "Could not load the checkpoint to the patch encoder model in strict way. "
                f"Received error: {e}. Loading in non-strict way."
            )
            if (
                self.model.patch_encoder.patch_embedding.weight.size()
                != model_checkpoint["patch_encoder.patch_embedding.weight"].size()
            ):
                logger.warning(
                    "The patch embedding weights size does not match the checkpoint size. "
                    "Removing patch embedding weights from the checkpoint."
                )
                model_checkpoint.pop("patch_encoder.patch_embedding.weight")
                model_checkpoint.pop("patch_encoder.patch_embedding.bias")
            if (
                self.model.patch_encoder.base.embeddings.position_embeddings.weight.size()
                != model_checkpoint[
                    "patch_encoder.base.embeddings.position_embeddings.weight"
                ].size()
            ):
                logger.warning(
                    "The position embeddings weights size does not match the checkpoint size. "
                    "Removing position embeddings weights from the checkpoint."
                    f"Expected sequence len in CLAMP2: 512, received: {self.sequence_length}"
                )
                model_checkpoint.pop(
                    "patch_encoder.base.embeddings.position_embeddings.weight"
                )
            self.model.load_state_dict(model_checkpoint, strict=False)

    def _download_checkpoint(self) -> None:
        logger.info(
            f"Downloading CLaMP2 weights from: {self.weights_url} into {self.weights_path}"
        )
        download_file(self.weights_url, self.weights_path, verbose=True)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if data.dim() == 3:
            masks = (data != 0).any(dim=-1).long()
        else:
            masks = (data != 0).long()

        hidden_states = self.model.get_music_features(
            music_inputs=data, music_masks=masks
        )

        mu = self.fc_mu(hidden_states)
        logvar = self.fc_logvar(hidden_states)

        return mu, logvar


class ClampEncoder(Encoder):
    def __init__(
        self,
        patchilizer_config: Dict,
        sequence_length: int,
        patch_size: int,
        latent_dim: int,
        vocab_size: int,
        model_name: str | None = None,
    ) -> None:
        super(ClampEncoder, self).__init__(
            latent_dim=latent_dim,
            name="clamp",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
        )
        self.device = get_possible_device()  #  TODO probbly to remove
        self.weights_path = base_weights_path / "clamp.bin"
        patchilizer_config = BertConfig(**patchilizer_config)
        self.patch_encoder = PatchEncoder(
            config=patchilizer_config,
            patch_size=patch_size,
            projection_layer=patchilizer_config.hidden_size,
            vocab_size=self.vocab_size,
        )
        self.model = CLaMPModel(patch_encoder=self.patch_encoder, latent_dim=latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.model_name = model_name

    def load_pretrained(self) -> None:
        raise NotImplementedError(
            "Think i there is possibility to use pretrained but for MIDI (so different input shape"
        )

    def _download_checkpoint(self) -> None:
        if self.model_name is None:
            raise ValueError("Model name is not provided for CLAMP encoder")
        if not self.weights_path.exists():
            logger.info(
                f"Downloading CLAMP V1 model from: {self.model_name} to local machine"
            )
            self.weights_path.mkdir(parents=True, exist_ok=True)
            config_url = f"https://huggingface.co/sander-wood/{self.model_name}/raw/main/config.json"
            model_url = f"https://huggingface.co/sander-wood/{self.model_name}/resolve/main/pytorch_model.bin"
            chunk_size = 1024 * 1024  # 1MB

            with requests.get(config_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(self.weights_path.parent / "clamp_config.json", "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading config",
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Download model file
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(self.weights_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading model",
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = (data != 0).any(dim=-1).long()

        hidden_states = self.model.get_music_features(
            music_inputs=data, music_masks=masks
        )

        mu = self.fc_mu(hidden_states)
        logvar = self.fc_logvar(hidden_states)

        return mu, logvar


def encoder_mapper(encoder_name: str) -> Type[Encoder]:
    if encoder_name == "clamp2":
        return Clamp2Encoder
    elif encoder_name == "clamp":
        return ClampEncoder
    elif encoder_name == "dummy":
        return DummyEncoder
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}, possible: ['simple_conv']")
