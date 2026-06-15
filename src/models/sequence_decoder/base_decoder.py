import logging as lg
from abc import ABC, abstractmethod

import torch
from torch import nn

from src.constants import Conditioning

logger = lg.getLogger(__name__)


class BaseDecoder(nn.Module, ABC):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        name: str,
        sequence_length: int,
        patch_size: int,
        bos_token: int,
        eos_token: int,
        pad_token: int,
        hidden_size: int,
        conditioning: str,
        class_conditioning: bool,
        dropout: float = 0.1,
        num_decoder_layers: int = 1,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        self.intermediate_ff_size = 4 * hidden_size
        self._num_classes = num_classes
        self._class_embedding = nn.Embedding(num_classes, hidden_size)
        self._class_conditioning = class_conditioning
        self._cache_params = None
        special_token_ids = [self.bos_token, self.eos_token, self.pad_token]
        self.register_buffer("special_ids", torch.tensor(special_token_ids, dtype=torch.long))
        if self.latent_dim is not None:
            self.latent_proj = nn.Linear(latent_dim, self.hidden_size)
        self._concat_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self._conditioning = Conditioning(conditioning)
        if self._class_conditioning and self.latent_dim:
            raise ValueError(
                "Class conditioning and latent conditioning cannot be enabled at the same time. Please choose one or the other."
            )
        self._concat_conditioning_layer = self._concat_proj if self.latent_dim else self._class_embedding
        logger.info(
            f"Initialize Decoder: {self.__class__.__name__} with conditioning: {self._conditioning} - Class Conditioning: {self._class_conditioning} - Latent Dim: {self.latent_dim}"
        )

    def _condition_input(
        self,
        token_embedding: torch.Tensor,
        conditioned_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if not self._class_conditioning and not self.latent_dim:
            raise ValueError(
                "At least one conditioning method (class or latent) must be enabled to use conditioning input method."
            )
        match self._conditioning:
            case Conditioning.ADD:
                return token_embedding + conditioned_embedding
            case Conditioning.CONCATENATION:
                concat_input = torch.cat([token_embedding, conditioned_embedding], dim=-1)  # [B, Seq_Len, Hidden*2]
                return self._concat_conditioning_layer(concat_input)  # [B, Seq_Len, Hidden]
            case _:
                raise ValueError(f"Unsupported conditioning method: {self._conditioning}")

    def _prepare_input(
        self,
        input_embeddings: torch.Tensor,
        latent_vector: torch.Tensor | None,
        class_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, current_sequence_len, _ = input_embeddings.shape

        if self._class_conditioning and self.latent_dim:
            raise ValueError(
                "Class conditioning and latent conditioning cannot be enabled at the same time. Please choose one or the other."
            )
        if self._class_conditioning:
            class_embedding = self._prepare_class_embedding(
                class_idx=class_idx,
                batch_size=batch_size,
                current_sequence_len=current_sequence_len,
            )
            return self._condition_input(token_embedding=input_embeddings, conditioned_embedding=class_embedding)
        elif self.latent_dim is not None:
            latent_proj_expanded = self._prepare_latent_embedding(
                latent_vector=latent_vector,
                batch_size=batch_size,
                current_sequence_len=current_sequence_len,
            )
            return self._condition_input(
                token_embedding=input_embeddings,
                conditioned_embedding=latent_proj_expanded,
            )
        else:
            return input_embeddings

    def _prepare_latent_embedding(
        self, latent_vector: torch.Tensor, batch_size: int, current_sequence_len: int
    ) -> torch.Tensor:
        if latent_vector is None:
            raise ValueError("Latent vector must be provided for latent conditioning.")
        latent_proj = self.latent_proj(latent_vector)
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(batch_size, current_sequence_len, self.hidden_size)
        return latent_proj_expanded

    def _prepare_class_embedding(
        self, class_idx: torch.Tensor | None, batch_size: int, current_sequence_len: int
    ) -> torch.Tensor:
        if class_idx is None:
            raise ValueError("Class index must be provided for class conditioning.")
        class_emb = self._class_embedding(class_idx)
        class_emb_expanded = class_emb.unsqueeze(1).expand(batch_size, current_sequence_len, self.hidden_size)
        return class_emb_expanded

    @abstractmethod
    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_layers_to_track(self):
        pass

    def clean_cache(self):
        self._cache_params = None

    @property
    def num_classes(self):
        return self._num_classes
