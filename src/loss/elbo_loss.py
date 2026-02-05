from typing import Tuple
from src.loss.base_vae_loss import BaseVaeLoss

import torch


class ElboLoss(BaseVaeLoss):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        turn_annealing: bool,
        annealing_steps: int,
        beta_max: float,
        kl_weight: float,
        free_bits: float,
    ) -> None:
        """
        ELBO Loss Function for Variational Autoencoders.

        :param vocab_size: Size of the vocabulary for reconstruction loss.
        :param turn: Whether to use annealing for the KL divergence term.
        :param annealing_steps: Number of steps over which to anneal the KL divergence weight.
        :param beta_max: Maximum weight for the KL divergence term during annealing.
        :param kl_weight: Fixed weight for the KL divergence term if not using annealing.
        """
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            turn_annealing=turn_annealing,
            annealing_steps=annealing_steps,
            beta_max=beta_max,
            kl_weight=kl_weight,
            free_bits=free_bits,
        )

    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        global_step: int,
        train: bool,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            org_data: Original input data.
            reconstruction: Reconstructed data from the model.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.
            global_step: Current training step.
            train: Boolean indicating if the model is in training mode.

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
            ignore_index=self._pad_token_id,
            reduction="mean",
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)

        annealing_beta = self._get_annealing_beta(global_step)

        kl_div_clamped = torch.max(
            kl_div, torch.tensor(self._free_bits).to(kl_div.device)
        )
        loss = recon_loss + annealing_beta * kl_div_clamped
        name = "train" if train else "val"

        return loss, {
            f"{name}_recon_loss": recon_loss,
            f"{name}_kl_div": kl_div,
            f"{name}_annealing_beta": annealing_beta,
        }
