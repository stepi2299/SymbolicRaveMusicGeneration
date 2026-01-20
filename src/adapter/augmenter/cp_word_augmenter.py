import torch
import random
from miditok import TokenizerConfig
import logging as lg

from src.adapter.augmenter.base_augmenter import BaseAugmenter
from src.constants import CP_WORD_INDEXES, CP_TOKEN_SIZE

logger = lg.getLogger(__name__)


class CpWordAugmenter(BaseAugmenter):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        augment_velocity: bool = False,
        augment_pitch: bool = False,
        velocity_interval: int = 0,
        pitch_interval: int = 0,
    ) -> None:
        self.tokenizer_config = tokenizer_config
        self._augment_velocity = augment_velocity
        self._augment_pitch = augment_pitch
        self._velocity_interval = velocity_interval
        self._pitch_interval = pitch_interval
        logger.info(
            f"CpWordAugmenter initialized with augment_pitch={augment_pitch}, augment_velocity={augment_velocity}, "
            f"max_pitch_interval={self._pitch_interval}, max_vel_interval={self._velocity_interval}"
        )

    def augment(self, music_data: torch.Tensor) -> torch.Tensor:
        if self._augment_velocity:
            shift_velocity = random.choice(
                [-self._velocity_interval, self._velocity_interval]
            )
            music_data = self._modify_one_family(
                music_sequence=music_data,
                family_name="velocity",
                min_range=4,
                shift_interval=shift_velocity,
            )
        if self._augment_pitch:
            shift_pitch = random.choice([-self._pitch_interval, self._pitch_interval])
            music_data = self._modify_one_family(
                music_sequence=music_data,
                family_name="pitch",
                min_range=4,
                shift_interval=shift_pitch,
            )
        return music_data

    @staticmethod
    def _modify_one_family(
        music_sequence: torch.Tensor,
        family_name: str,
        min_range: int,
        shift_interval: int,
    ) -> torch.Tensor:
        augmented = music_sequence.clone()
        family_tokens = augmented[:, CP_WORD_INDEXES[family_name]]
        mask = family_tokens >= min_range
        new_family_tokens = family_tokens[mask] + shift_interval
        new_family_tokens = torch.clamp(
            new_family_tokens, min=min_range, max=CP_TOKEN_SIZE[family_name]
        )

        augmented[mask, CP_WORD_INDEXES[family_name]] = new_family_tokens
        return augmented
