import torch
import logging as lg

from transformers import TransfoXLConfig, TransfoXLModel
from src.models.sequence_decoder.base_decoder import BaseDecoder

logger = lg.getLogger(__name__)


class TransformerXL(BaseDecoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        mem_len: int,
        dropout: float,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        attention_dropout: float,
        conditioning: str,
        class_conditioning: bool,
        num_classes: int,
    ) -> None:
        super(TransformerXL, self).__init__(
            latent_dim=latent_dim,
            name="transformer_xl",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            conditioning=conditioning,
            class_conditioning=class_conditioning,
            num_classes=num_classes,
        )

        config = TransfoXLConfig(
            d_model=self.hidden_size,
            n_head=self.num_heads,
            n_layer=num_decoder_layers,
            d_inner=self.intermediate_ff_size,
            vocab_size=self.vocab_size,
            mem_len=mem_len,  # Length of the memory
            dropout=self.dropout,
            dropatt=attention_dropout,
            # Important! Disable adaptive input and output embeddings
            adaptive=False,
            div_val=1,  # No division of embedding size,
            pre_lnorm=True,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
        )

        self.model = TransfoXLModel(config)

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(
            input_embedding, latent_vector, class_idx
        )

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds, mems=self._cache_params
        )
        self._cache_params = outputs.mems
        return outputs.last_hidden_state  # [B, Seq_Len, Hidden]

    def get_layers_to_track(self):
        return self.model.layers
