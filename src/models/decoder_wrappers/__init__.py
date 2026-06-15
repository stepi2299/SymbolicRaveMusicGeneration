from omegaconf import DictConfig, OmegaConf

from src.models.decoder_wrappers.base_wrapper import BaseWrapper
from src.models.decoder_wrappers.cp_wraper import CompoundWordWrapper
from src.models.decoder_wrappers.patch_wrapper import PatchWrapper
from src.models.decoder_wrappers.simple_wrapper import SimpleWrapper
from src.models.sequence_decoder import BaseDecoder


def get_decoder_wrapper(wrapper_type: str, decoder: BaseDecoder, params: DictConfig) -> BaseWrapper:
    match wrapper_type.lower():
        case "simple":
            return SimpleWrapper(decoder=decoder, **params)
        case "compound_word":
            return CompoundWordWrapper(decoder=decoder, **params)
        case "patch":
            return PatchWrapper(decoder=decoder, **params)
        case _:
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}, supported: simple, compound_word, patch")
