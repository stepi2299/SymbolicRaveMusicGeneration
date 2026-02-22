import torch

import logging as lg
from transformers import GPTNeoXConfig, GPTNeoXModel
from src.models.utils.custom_rope import CustomGPTNeoXModel

from src.models.sequence_decoder.base_decoder import BaseDecoder


logger = lg.getLogger(__name__)


class TransformerRope(BaseDecoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        rotary_pct: float,
        rotary_emb_base: int,
        conditioning: str,
        custom: bool,
        use_cross_attention: bool,
        class_conditioning: bool,
        num_classes: int,
    ) -> None:
        super(TransformerRope, self).__init__(
            latent_dim=latent_dim,
            name="transformer_rope",
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

        config = GPTNeoXConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_decoder_layers,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=vocab_size,
            classifier_dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=sequence_length,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )
        self.custom = custom
        if custom:
            self.model = CustomGPTNeoXModel(
                config, use_cross_attention=use_cross_attention
            )
        else:
            self.model = GPTNeoXModel(config)

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, current_sequence_len, _ = input_embedding.shape
        if self.custom:
            if self._class_conditioning and self.latent_dim is not None:
                raise ValueError(
                    "Its not possible to use class bot - conditioning with latent vector projection in TransformerRope decoder. Please set latent_dim to None or _class_conditioning to False."
                )
            if self._class_conditioning:
                conditioning_embedding = self._prepare_class_embedding(
                    class_idx, batch_size, current_sequence_len
                )
            elif self.latent_dim is not None:
                conditioning_embedding = self._prepare_latent_embedding(
                    latent_vector, batch_size, current_sequence_len
                )
            else:
                raise ValueError(
                    "Either class conditioning or latent vector projection must be enabled for the custom TransformerRope decoder."
                )
            outputs = self.model(
                inputs_embeds=input_embedding,
                encoder_hidden_states=conditioning_embedding,
                past_key_values=self._cache_params,
                return_dict=True,
                use_cache=use_cache,
            )
        else:
            input_embedding = self._prepare_input(
                input_embedding, latent_vector, class_idx
            )
            outputs = self.model(
                inputs_embeds=input_embedding,
                past_key_values=self._cache_params,
                return_dict=True,
                use_cache=use_cache,
            )
        if use_cache:
            self._cache_params = outputs.past_key_values

        return outputs.last_hidden_state

    def get_layers_to_track(self):
        return self.model.layers
