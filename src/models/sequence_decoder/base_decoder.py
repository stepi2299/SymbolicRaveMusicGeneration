import torch
from torch import nn
from abc import ABC, abstractmethod
import logging as lg

from src.constants import Conditioning

logger = lg.getLogger(__name__)


class BaseDecoder(nn.Module, ABC):
    def __init__(
        self,
        vocab_size: int,
        name: str,
        sequence_length: int,
        patch_size: int,
        bos_token: int,
        eos_token: int,
        pad_token: int,
        hidden_size: int,
        conditioning: str,
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
        self._cache_params = None
        special_token_ids = [self.bos_token, self.eos_token, self.pad_token]
        self.register_buffer(
            "special_ids", torch.tensor(special_token_ids, dtype=torch.long)
        )
        if self.latent_dim is not None:
            self.latent_proj = nn.Linear(latent_dim, self.hidden_size)
        self._concat_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self._conditioning = Conditioning(conditioning)
        logger.info(
            f"Initialize Decoder: {self.__class__.__name__} with conditioning: {self._conditioning}"
        )

    def _condition_input(
        self,
        token_embedding: torch.Tensor,
        latent_projection: torch.Tensor,
        class_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO: Add class_idx conditioning for FiLM or other methods if needed
        match self._conditioning:
            case Conditioning.ADD:
                return token_embedding + latent_projection
            case Conditioning.CONCATENATION:
                concat_input = torch.cat(
                    [token_embedding, latent_projection], dim=-1
                )  # [B, Seq_Len, Hidden*2]
                return self._concat_proj(concat_input)  # [B, Seq_Len, Hidden]
            case _:
                raise ValueError(
                    f"Unsupported conditioning method: {self._conditioning}"
                )

    def _prepare_input(
        self,
        input_embeddings: torch.Tensor,
        latent_vector: torch.Tensor | None,
        class_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, current_sequence_len = input_embeddings.shape

        # [B, Latent] -> [B, Hidden]
        latent_proj = self.latent_proj(latent_vector)

        # [B, 1, Hidden] -> [B, Seq_Len, Hidden]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        return self._condition_input(
            token_embedding=input_embeddings, latent_projection=latent_proj_expanded
        )

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
