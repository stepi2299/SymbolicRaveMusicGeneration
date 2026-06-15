import logging as lg

import torch
from transformers import MistralConfig, MistralModel

from src.models.sequence_decoder.base_decoder import BaseDecoder

logger = lg.getLogger(__name__)


class MistralDecoder(BaseDecoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        attention_dropout: float,
        rope_theta: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        sliding_window: int,
        num_key_value_heads: int,
        conditioning: str,
        class_conditioning: bool,
        num_classes: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name="mistral_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            hidden_size=hidden_size,
            dropout=attention_dropout,
            num_decoder_layers=num_decoder_layers,
            conditioning=conditioning,
            class_conditioning=class_conditioning,
            num_classes=num_classes,
        )
        config = MistralConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_decoder_layers,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=vocab_size,
            max_position_embeddings=self.sequence_length,
            attention_dropout=attention_dropout,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
        )
        self.model = MistralModel(config)

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(input_embedding, latent_vector, class_idx)

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=self._cache_params,
            return_dict=True,
            use_cache=use_cache,
        )
        if use_cache:
            self._cache_params = outputs.past_key_values
        return outputs.last_hidden_state

    def get_layers_to_track(self):
        return self.model.layers
