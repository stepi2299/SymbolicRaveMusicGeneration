import torch
from torch import nn


class PatchEmbeddings(nn.Module):
    """
    Goal of this class is project sequence of patches into [Batch, Sequence Length (Num Patch), Hidden Size] sized
    projection vector of input patches
    """

    def __init__(
        self,
        vocab_size: int,
        patch_size: int,
        token_embedding_size: int,
        hidden_size: int,
        dropout: int = 0,
    ):
        super().__init__()
        self._single_token_embedding = nn.Embedding(vocab_size, token_embedding_size)
        self._flattened_dim = token_embedding_size * patch_size
        self._patch_projection = nn.Sequential(
            nn.Linear(self._flattened_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        :param patches: Sequence of patches in dimension [Batch, Sequence Length (Num Patch), Patch]
        :return: tensor in dimension [Batch, Sequence Length (Num Patch), Hidden Dimension (Projection Layer)]
        """
        batch_size, current_patch_num, patch_size = patches.shape
        tok_emb = self._single_token_embedding(patches)  # [B, current_patch_num, patch_size, token embedding]
        tok_emb_flat = tok_emb.view(batch_size, current_patch_num, -1)  # [B, patch_num, patch_size * token embedding]
        return self._patch_projection(tok_emb_flat)  # [B, patch_num, hidden size]


class PatchOutputHead(nn.Module):
    """
    Goal of this class is create output for training purpose in shape
    [Batch, Sequence Length (Num Patch), Patch Size, Vocab size] or for generation purpose enable transforming it into
    [Batch, Sequence Length (Num Patch), Patch Size] with Best Id from vocab size
    """

    def __init__(
        self,
        vocab_size: int,
        patch_size: int,
        hidden_size: int,
        intermediate_layer: int | None = None,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._patch_size = patch_size
        self._intermediate_layer = intermediate_layer if intermediate_layer else self._hidden_size * 4

        self.patch_head = nn.Sequential(
            nn.Linear(self._hidden_size, self._intermediate_layer),
            nn.GELU(),
            nn.LayerNorm(self._intermediate_layer),
            nn.Linear(self._intermediate_layer, patch_size * vocab_size),
        )

    def forward(self, hidden_output: torch.Tensor) -> torch.Tensor:
        flattened_patch = self.patch_head(hidden_output)
        restructured_patch = flattened_patch.unflatten(
            dim=-1, sizes=(self._patch_size, self._vocab_size)
        )  # [B, Seq Len, Patch Size, Vocab Size]
        return restructured_patch
