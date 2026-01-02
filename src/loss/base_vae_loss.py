from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseVaeLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            org_data: Original input data.
            reconstruction: Reconstructed data from the model.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.
            global_step: Current training step.

        Returns:
            loss: The computed loss.
            metrics: A dictionary of additional metrics to log.
        """
        pass
