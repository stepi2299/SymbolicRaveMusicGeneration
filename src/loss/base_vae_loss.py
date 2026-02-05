from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseVaeLoss(ABC):
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

    @abstractmethod
    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        global_step: int,
        train: bool,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            org_data: Original input data.
            reconstruction: Reconstructed data from the model.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.
            global_step: Current training step.
            train: Boolean indicating if the model is in training mode.

        Returns:
            loss: The computed loss.
            metrics: A dictionary of additional metrics to log.
        """
        pass

    def _get_annealing_beta(self, global_step: int) -> float:
        if self._turn_annealing:
            return min(1.0, global_step / self._annealing_steps) * self._beta_max
        else:
            return self._kl_weight
