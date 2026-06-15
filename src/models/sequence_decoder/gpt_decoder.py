import logging as lg

import torch
from torch import nn

from src.models.sequence_decoder.base_decoder import BaseDecoder

logger = lg.getLogger(__name__)


class GptDecoder(BaseDecoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        dropout: float,
        conditioning: str,
        class_conditioning: bool,
        num_classes: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name="gpt",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=dropout,
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            conditioning=conditioning,
            class_conditioning=class_conditioning,
            num_classes=num_classes,
        )

        self.positional_embeddings = nn.Embedding(self.sequence_length, self.hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.intermediate_ff_size,
            dropout=self.dropout,
            norm_first=True,
        )
        self.model = nn.TransformerDecoder(layer, num_layers=self.num_decoder_layers)

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Getting current sequence length (for generation when len is growing)
        batch_size, current_sequence_len, _ = input_embedding.shape

        position_idx = torch.arange(current_sequence_len, dtype=torch.long, device=input_embedding.device)
        position_idx = position_idx.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.positional_embeddings(position_idx)  # [1, seq_len, hidden]

        decoder_input = self._prepare_input(input_embedding, latent_vector, class_idx)
        decoder_input = decoder_input + pos_emb
        decoder_input = decoder_input.permute(1, 0, 2)

        mask = nn.Transformer.generate_square_subsequent_mask(current_sequence_len).to(input_embedding.device)

        return self.model(
            decoder_input,
            memory=decoder_input,
            tgt_mask=mask,
        ).transpose(0, 1)

    def get_layers_to_track(self):
        return self.model.layers
