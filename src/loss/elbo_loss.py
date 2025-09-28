from typing import Tuple
from src.loss.base_vae_loss import BaseVaeLoss

import torch


class ElboLoss(BaseVaeLoss):
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            org_data: Original input data.
            reconstruction: Reconstructed data from the model.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.

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
            ),  # Reshape to (batch*patch_count, vocab_size)
            org_data.view(-1),  # Reshape to (batch*patch_count)
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)
        loss = recon_loss + kl_div

        return loss, {"recon_loss": recon_loss, "kl_div": kl_div}
