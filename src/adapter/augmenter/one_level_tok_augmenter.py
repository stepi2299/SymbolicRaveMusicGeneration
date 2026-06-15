import logging as lg
import random

import torch
from miditok import TokenizerConfig

from src.adapter.augmenter.base_augmenter import BaseAugmenter

logger = lg.getLogger(__name__)


class OneLevelTokAugmenter(BaseAugmenter):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        augment_velocity: bool = False,
        augment_pitch: bool = False,
        velocity_interval: int = 0,
        pitch_interval: int = 0,
    ) -> None:
        self._tokenizer_config = tokenizer_config
        self._augment_velocity = augment_velocity
        self._augment_pitch = augment_pitch
        self._pitch_min = 4
        self._pitch_max = self._pitch_min + (tokenizer_config.pitch_range[1] - tokenizer_config.pitch_range[0])
        self._velocity_min = self._pitch_max + 1
        self._velocity_max = self._velocity_min + tokenizer_config.num_velocities
        self._max_vel_interval = velocity_interval
        self._max_pitch_interval = pitch_interval
        logger.info(
            f"OneLevelTokAugmenter initialized with augment_pitch={augment_pitch}, augment_velocity={augment_velocity}, "
            f"pitch_range=({self._pitch_min}, {self._pitch_max}), velocity_range=({self._velocity_min}, {self._velocity_max}), "
            f"max_pitch_interval={self._max_pitch_interval}, max_vel_interval={self._max_vel_interval}"
        )

    def augment(self, music_data: torch.Tensor) -> torch.Tensor:
        if self._augment_pitch:
            pitch_interval = random.randint(-self._max_pitch_interval, self._max_pitch_interval)
            music_data = self._modify_one_family(
                music_sequence=music_data,
                min_range=self._pitch_min,
                max_range=self._pitch_max,
                shift_interval=pitch_interval,
            )
        if self._augment_velocity:
            velocity_interval = random.randint(-self._max_vel_interval, self._max_vel_interval)
            music_data = self._modify_one_family(
                music_sequence=music_data,
                min_range=self._velocity_min,
                max_range=self._velocity_max,
                shift_interval=velocity_interval,
            )
        return music_data

    @staticmethod
    def _modify_one_family(
        music_sequence: torch.Tensor,
        min_range: int,
        max_range: int,
        shift_interval: int,
    ) -> torch.Tensor:
        mask = (music_sequence >= min_range) & (music_sequence <= max_range)
        shifted_values = music_sequence[mask] + shift_interval
        music_sequence[mask] = torch.clamp(shifted_values, min=min_range, max=max_range)
        return music_sequence
