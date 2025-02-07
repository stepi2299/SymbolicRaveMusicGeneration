from __future__ import annotations

import torch
from transformers import BertConfig, PreTrainedModel, BertModel


class PatchEncoder(PreTrainedModel):
    def __init__(
        self, config: BertConfig, patch_size: int, projection_layer: int
    ) -> None:
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(patch_size * 128, projection_layer)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.config = config
        self.base = BertModel(config=config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        self.patch_size = patch_size
        self.projection_layer = projection_layer

    def forward(
        self,
        input_patches: torch.Tensor,  # [batch_size, seq_length, hidden_size]
        input_masks: torch.Tensor,  # [batch_size, seq_length]
    ) -> torch.Tensor:
        # Transform input_patches into embeddings
        input_patches = torch.nn.functional.one_hot(input_patches, num_classes=128)
        input_patches = input_patches.reshape(
            len(input_patches), -1, self.patch_size * 128
        ).type(torch.FloatTensor)
        input_patches = self.patch_embedding(input_patches.to(self.device))

        # Apply BERT model to input_patches and input_masks
        return self.base(inputs_embeds=input_patches, attention_mask=input_masks)


class PatchDecoder(PreTrainedModel):
    """
    Inverse of PatchEncoder: It takes in BERT-friendly embeddings (hidden states)
    and projects them back to patch-space (patch_size * 128).
    """

    def __init__(
        self,
        config: BertConfig,
        patch_size: int,
        projection_layer: int,
        vocab_size: int = 128,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.patch_size = patch_size
        self.projection_layer = projection_layer
        self.vocab_size = vocab_size

        # "Decode" backbone (like a BERT model).
        self.base = BertModel(config)

        # Inverse patch embedding:
        #   from 'projection_layer' -> 'patch_size * 128'
        self.patch_embedding_inv = torch.nn.Linear(
            projection_layer, patch_size * vocab_size
        )
        torch.nn.init.normal_(self.patch_embedding_inv.weight, std=0.02)

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [batch_size, seq_length, projection_layer]
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            inputs_embeds: The 'decoded' embeddings to pass through BERT.
            attention_mask: Optional, same usage as in BERT.
        Returns:
            patch_logits: [batch_size, seq_length, patch_size * 128]
        """

        # Run "decoder" BERT
        decoder_outputs = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Take BERT’s last hidden states
        hidden_states = (
            decoder_outputs.last_hidden_state
        )  # [batch_size, seq_length, hidden_size]

        # Project back to patch space (logits)
        patch_logits = self.patch_embedding_inv(
            hidden_states
        )  # [batch_size, seq_length, patch_size*vocab_size]

        return patch_logits


class CLaMPModel(PreTrainedModel):
    def __init__(self, patch_encoder: PatchEncoder, latent_dim: int) -> None:
        super().__init__(patch_encoder.config)
        self.patch_encoder = patch_encoder
        self.music_proj = torch.nn.Linear(
            self.patch_encoder.projection_layer, latent_dim
        )  # Linear layer for music projections
        torch.nn.init.normal_(
            self.music_proj.weight, std=0.02
        )  # Initialize weights with normal distribution

    def get_music_features(
        self, music_inputs: torch.Tensor, music_masks: torch.Tensor
    ) -> torch.Tensor:
        music_features = self.patch_encoder(
            music_inputs.to(self.device), music_masks.to(self.device)
        )["last_hidden_state"]

        # Normalize features (Reduce Temporal Dimension)
        music_features = self._avg_pooling(music_features, music_masks)
        music_features = self.music_proj(music_features)

        return music_features

    def _avg_pooling(
        self, input_features: torch.FloatTensor, input_masks: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Applies average pooling to the input features.

        Args:
            input_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, hidden_size)`):
                Tensor containing the input features.
            input_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`):
                Tensor containing the attention masks for the input features.

        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`:
                The pooled features.
        """
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1)

        return avg_pool


class CLaMPDecoderModel(PreTrainedModel):
    """
    Mirrors CLaMP2Model but goes from latent space -> patch decoder.
    """

    def __init__(self, patch_decoder: PatchDecoder, latent_dim: int):
        super().__init__(patch_decoder.config)
        self.patch_decoder = patch_decoder
        self.latent_dim = latent_dim

        # Project the latent vector to patch_decoder's projection_layer size
        self.latent_proj = torch.nn.Linear(latent_dim, patch_decoder.projection_layer)
        torch.nn.init.normal_(self.latent_proj.weight, std=0.02)

    def expand_latent(
        self, latent_vectors: torch.Tensor, seq_length: int
    ) -> torch.Tensor:
        """
        Expand a batch of latent vectors [batch_size, latent_dim]
        into a sequence of embeddings [batch_size, seq_length, projection_layer].
        One simple approach is to tile the latent vector across seq_length.
        """
        # Project latent to the same dimension as patch_decoder expects
        # i.e., `projection_layer` dimension
        batch_size = latent_vectors.size(0)
        projected = self.latent_proj(latent_vectors)  # [batch_size, projection_layer]

        # Tile across seq_length
        # shape -> [batch_size, seq_length, projection_layer]
        projected = projected.unsqueeze(1).expand(batch_size, seq_length, -1)
        return projected

    def forward(
        self,
        latent_vectors: torch.Tensor,  # [batch_size, latent_dim]
        seq_length: int,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Goes from [batch_size, latent_dim] -> [batch_size, seq_length, patch_size*vocab_size]
        """
        # Expand latent vector to match the sequence dimension
        inputs_embeds = self.expand_latent(latent_vectors, seq_length)

        # Decode to patch logits
        patch_logits = self.patch_decoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        return patch_logits
