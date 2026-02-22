from typing import Type
from omegaconf import DictConfig
from src.adapter import MusicAdapter

from src.models.sequence_decoder.mistral_decoder import MistralDecoder
from src.models.sequence_decoder.mamba_decoder import MambaDecoder
from src.models.sequence_decoder.transformer_rope import TransformerRope
from src.models.sequence_decoder.transformer_xl import TransformerXL
from src.models.sequence_decoder.gpt_decoder import GptDecoder
from src.models.sequence_decoder.base_decoder import BaseDecoder


def _decoder_mapper(decoder_name: str) -> Type[BaseDecoder]:
    if decoder_name == "gpt_decoder":
        return GptDecoder
    elif decoder_name == "transformer_xl":
        return TransformerXL
    elif decoder_name == "transformer_rope":
        return TransformerRope
    elif decoder_name == "mistral":
        return MistralDecoder
    elif decoder_name == "mamba":
        return MambaDecoder
    else:
        raise ValueError(
            f"Unknown decoder: {decoder_name}, possible: gpt_decoder, transformer_xl, transformer_rope, mistral, mamba"
        )


def get_decoder(
    decoder_cfg: DictConfig, adapter: MusicAdapter, latent_dim: int | None
) -> BaseDecoder:
    return _decoder_mapper(decoder_cfg.name)(
        latent_dim=latent_dim,
        vocab_size=adapter.vocab_size,
        sequence_length=adapter.sequence_length,
        patch_size=adapter.patch_size,
        pad_token=adapter.pad_token_id,
        bos_token=adapter.bos_token_id,
        eos_token=adapter.eos_token_id,
        **decoder_cfg.params,
    )


__all__ = ["get_decoder", "BaseDecoder"]
