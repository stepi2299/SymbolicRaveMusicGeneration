import torch
import torch.nn as nn
import logging as lg

from src.constants import CP_WORD_INDEXES

logger = lg.getLogger(__name__)


class CompoundWordEmbedding(nn.Module):
    """
    Realizuje "Combining Token Embeddings of Adaptive Sizes".
    Każdy atrybut (np. pitch, velocity) ma własny embedding o dopasowanym rozmiarze,
    który jest następnie rzutowany do wymiaru modelu i sumowany.
    """

    def __init__(self, token_types: dict[str, int], hidden_size: int):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self._hidden_size = hidden_size

        if len(token_types) != len(CP_WORD_INDEXES):
            raise ValueError(
                f"Token types provided ({len(token_types)}) do not match expected number of CP Word attributes ({len(CP_WORD_INDEXES)})."
            )

        for name, vocab_size in token_types.items():
            if name not in CP_WORD_INDEXES:
                raise ValueError(
                    f"Attribute name '{name}' is not recognized. Expected attributes are: {list(CP_WORD_INDEXES.keys())}"
                )
            # Adaptive Size: Heurystyka - pierwiastek z vocab_size lub sztywny limit.
            embed_dim = min(hidden_size, int(vocab_size**0.5) * 16)
            embed_dim = max(32, embed_dim)  # Minimum 32 dim

            self.embeddings[name] = nn.Embedding(vocab_size, embed_dim)
            self.projections[name] = nn.Linear(embed_dim, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: [Batch, Seq_Len, Num_Attributes]
        # input_keys: lista nazw atrybutów w kolejności zgodnej z ostatnim wymiarem inputs

        batch_size, seq_len, _ = inputs.shape
        # Inicjalizacja zerami
        combined_embedding = torch.zeros(
            batch_size,
            seq_len,
            self._hidden_size,
            device=inputs.device,
        )

        for key_name, index in CP_WORD_INDEXES.items():
            attr_indices = inputs[:, :, index]  # [B, L]

            vocab_limit = self.embeddings[key_name].num_embeddings
            max_val_in_batch = attr_indices.max().item()
            min_val_in_batch = attr_indices.min().item()

            if max_val_in_batch >= vocab_limit or min_val_in_batch < 0:
                logger.error(f"Attribute: '{key_name}'")
                logger.error(f"Embedding Size (Vocab Limit): {vocab_limit}")
                logger.error(f"Found Index in Data: {max_val_in_batch}")
                logger.error(f"Min Index in Data: {min_val_in_batch}")
                raise ValueError(
                    f"Index {max_val_in_batch} out of bounds for attribute '{key_name}' (limit {vocab_limit})"
                )

            # Embed & Project
            emb = self.embeddings[key_name](attr_indices)  # [B, L, embed_dim]
            proj = self.projections[key_name](emb)  # [B, L, hidden_size]

            # Sumowanie (zgodnie z artykułem CP Transformer)
            combined_embedding = combined_embedding + proj

        return combined_embedding


class MultiHeadOutput(nn.Module):
    """
    Realizuje "Multi-head Output Module".
    Zamiast jednego wielkiego Softmaxa, mamy osobne klasyfikatory dla każdego atrybutu.
    """

    def __init__(self, hidden_size: int, token_types: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, vocab_size in token_types.items():
            self.heads[name] = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        # hidden_states: [Batch, Seq_Len, Hidden]
        outputs = {}
        for name, layer in self.heads.items():
            outputs[name] = layer(
                hidden_states
            )  # [Batch, Seq_Len, Vocab_Size_Attribute]
        return outputs
