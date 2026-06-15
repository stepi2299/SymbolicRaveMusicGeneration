from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
    ):
        self._vocab_size = vocab_size
        self._pad_token_id = pad_token_id

    @abstractmethod
    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        global_step: int,
        train: bool,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
