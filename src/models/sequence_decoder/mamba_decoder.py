import logging as lg

import torch
from transformers import MambaConfig, MambaModel

from src.models.sequence_decoder.base_decoder import BaseDecoder

logger = lg.getLogger(__name__)


class MambaDecoder(BaseDecoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        state_size: int,
        num_decoder_layers: int,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        expand: int,
        conv_kernel: int,
        conditioning: str,
        class_conditioning: bool,
        num_classes: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name="mamba_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            conditioning=conditioning,
            class_conditioning=class_conditioning,
            num_classes=num_classes,
        )
        config = MambaConfig(
            hidden_size=self.hidden_size,
            state_size=state_size,
            num_hidden_layers=self.num_decoder_layers,
            vocab_size=vocab_size,
            expand=expand,
            conv_kernel=conv_kernel,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )
        self.model = MambaModel(config)

    def forward(
        self,
        input_embedding: torch.Tensor,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(input_embedding, latent_vector=latent_vector, class_idx=class_idx)

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            cache_params=self._cache_params,
            use_cache=use_cache,
            return_dict=True,
        )
        if use_cache:
            self._cache_params = outputs.cache_params
        return outputs.last_hidden_state

    def get_layers_to_track(self):
        return self.model.layers
