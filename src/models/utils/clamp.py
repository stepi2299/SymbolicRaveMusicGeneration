from __future__ import annotations

import torch
from transformers import BertConfig, BertModel, PreTrainedModel


class PatchEncoder(PreTrainedModel):
    def __init__(
        self,
        config: BertConfig,
        patch_size: int,
        projection_layer: int,
        vocab_size: int,
    ) -> None:
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(patch_size * vocab_size, projection_layer)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.config = config
        self.base = BertModel(config=config)
        self.patch_size = patch_size
        self.projection_layer = projection_layer
        self.vocab_size = vocab_size

    def forward(
        self,
        input_patches: torch.Tensor,  # [batch, seq_len, patch_size] | [batch, seq_len] | [batch, seq_len, vocab_size] (float)
        input_masks: torch.Tensor,  # [batch_size, sequence_length]
    ) -> torch.Tensor:  # [batch_size, sequence_length, projection_layer]
        if input_patches.is_floating_point():
            # Soft input (e.g. Gumbel-Softmax one-hot): [batch, T, V]
            # Reshape to [batch, T/patch_size, patch_size * vocab_size] for patch_embedding
            input_patches = input_patches.reshape(len(input_patches), -1, self.patch_size * self.vocab_size)
        else:
            # Integer token input: one-hot encode first
            input_patches = torch.nn.functional.one_hot(
                input_patches, num_classes=self.vocab_size
            )  # → [batch, seq_len, patch_size, vocab_size] or [batch, seq_len, vocab_size]
            if len(input_patches.shape) != 3:
                input_patches = input_patches.reshape(
                    len(input_patches), -1, self.patch_size * self.vocab_size
                )  # [batch, seq_len, patch_size * vocab_size]
            else:
                input_patches = input_patches.reshape(
                    len(input_patches), -1, self.vocab_size
                )  # [batch, seq_len, vocab_size]  (patch_size == 1)
            input_patches = input_patches.float()  # keep on current device

        input_patches = self.patch_embedding(input_patches.to(self.device))

        # Apply BERT model to input_patches and input_masks
        return self.base(inputs_embeds=input_patches, attention_mask=input_masks)


class CLaMPModel(PreTrainedModel):
    def __init__(self, patch_encoder: PatchEncoder, latent_dim: int) -> None:
        super().__init__(patch_encoder.config)
        self.patch_encoder = patch_encoder
        self.music_proj = torch.nn.Linear(
            self.patch_encoder.projection_layer, latent_dim
        )  # Linear layer for music projections
        torch.nn.init.normal_(self.music_proj.weight, std=0.02)  # Initialize weights with normal distribution

    def get_music_features(self, music_inputs: torch.Tensor, music_masks: torch.Tensor) -> torch.Tensor:
        music_features = self.patch_encoder(music_inputs.to(self.device), music_masks.to(self.device))[
            "last_hidden_state"
        ]

        # Normalize features (Reduce Temporal Dimension)
        music_features = self._avg_pooling(music_features, music_masks)
        music_features = self.music_proj(music_features)

        return music_features

    def _avg_pooling(self, input_features: torch.FloatTensor, input_masks: torch.LongTensor) -> torch.FloatTensor:
        """
        Applies average pooling to the input features.

        Args:
            input_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Tensor containing the input features.
            input_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Tensor containing the attention masks for the input features.

        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`:
                The pooled features.
        """
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1)

        return avg_pool
