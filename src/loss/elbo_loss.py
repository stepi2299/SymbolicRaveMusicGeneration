from typing import Tuple
from src.loss.base_vae_loss import BaseVaeLoss

import torch


class ElboLoss(BaseVaeLoss):
    def __init__(
        self, vocab_size: int, turn: bool, annealing_steps: int, beta_max: float
    ) -> None:
        """
        ELBO Loss Function for Variational Autoencoders.

        :param vocab_size: Size of the vocabulary for reconstruction loss.
        :param turn: Whether to use annealing for the KL divergence term.
        :param annealing_steps: Number of steps over which to anneal the KL divergence weight.
        :param beta_max: Maximum weight for the KL divergence term during annealing.
        """
        self._vocab_size = vocab_size
        self._annealing_steps = annealing_steps
        self._beta_max = beta_max
        self._turn = turn

    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            org_data: Original input data.
            reconstruction: Reconstructed data from the model.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.
            global_step: Current training step.

        Returns:
            loss: The computed ELBO loss which is the sum of reconstruction loss and KL divergence.
            metrics: A dictionary of additional metrics to log, including 'recon_loss' and 'kl_div'.

        recon_loss is computed using cross-entropy loss between the original data and the reconstruction.
        kl_div is computed as the KL divergence between the learned latent distribution and a standard normal distribution

        Elbo explanation in documentation
        """
        recon_loss = torch.nn.functional.cross_entropy(
            reconstruction.view(
                -1, self._vocab_size
            ),  # Reshape to (batch*sequence_length, vocab_size)
            org_data.view(-1),  # Reshape to (batch*sequence_length)
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)

        if self._turn:
            annealing_beta = (
                min(1.0, global_step / self._annealing_steps) * self._beta_max
            )
        else:
            annealing_beta = 1
        loss = recon_loss + annealing_beta * kl_div

        return loss, {
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "annealing_beta": annealing_beta,
        }
