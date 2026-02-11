from abc import ABC, abstractmethod
from typing import Tuple

import torch

from src.loss.base_loss import BaseLoss


class BaseVaeLoss(BaseLoss):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        turn_annealing: bool,
        annealing_steps: int,
        beta_max: float,
        kl_weight: float,
        free_bits: float = 0.0,
    ):
        self._kl_weight = kl_weight
        self._turn_annealing = turn_annealing
        self._annealing_steps = annealing_steps
        self._beta_max = beta_max
        self._vocab_size = vocab_size
        self._pad_token_id = pad_token_id
        self._free_bits = free_bits

    def _get_annealing_beta(self, global_step: int) -> float:
        if self._turn_annealing:
            return min(1.0, global_step / self._annealing_steps) * self._beta_max
        else:
            return self._kl_weight
