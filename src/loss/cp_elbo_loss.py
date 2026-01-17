import torch
import torch.nn as nn
from src.loss.base_vae_loss import BaseVaeLoss
from src.constants import CP_WORD_INDEXES


class CpElboLoss(BaseVaeLoss):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        kl_weight: float,
        attribute_weights: dict,
        turn_annealing: bool,
        annealing_steps: int,
        beta_max: float,
    ) -> None:
        super().__init__(
            annealing_steps=annealing_steps,
            beta_max=beta_max,
            turn_annealing=turn_annealing,
            kl_weight=kl_weight,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
        )

        self.attr_weights = attribute_weights
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="mean")

    def __call__(
        self,
        org_data: torch.Tensor,  # Prawdziwe dane: [B, L, Num_Attrs]
        reconstruction: dict[
            str, torch.Tensor
        ],  # Wyjście z modelu: {'pitch': [B, L, 128], ...}
        mu: torch.Tensor,  # Średnia z Encodera VAE [B, Latent]
        logvar: torch.Tensor,  # Logariytm wariancji z Encodera VAE [B, Latent]
        global_step: int,
        train: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            org_data: Original input data of shape [Batch, Seq_Len, Num_Attrs].
            reconstruction: A dictionary where each key is an attribute name and the value is
                            the reconstructed logits tensor of shape [Batch, Seq_Len, Vocab_Size].
            mu: Mean of the latent variable distribution of shape [Batch, Latent_Dim].
            logvar: Log variance of the latent variable distribution of shape [Batch, Latent_Dim].
            global_step: Current training step.
            train: Boolean indicating if the model is in training mode.
        Returns:
            loss: The computed Compound Word VAE loss which is the sum of weighted reconstruction loss and KL divergence.
            metrics: A dictionary of additional metrics to log, including 'recon_loss', 'kl_div', and 'annealing_beta'.
        """

        total_recon_loss = 0.0

        for attr_name, index in CP_WORD_INDEXES.items():
            # Pobieramy predykcje dla danego atrybutu
            # logits: [Batch, Seq_Len, Vocab_Size] -> spłaszczamy do [Batch*Seq_Len, Vocab]
            attr_logits = reconstruction[attr_name]
            B, L, V = attr_logits.shape
            attr_logits_flat = attr_logits.view(-1, V)

            # Pobieramy prawdziwe wartości dla tego atrybutu (i-ta kolumna targets)
            # targets: [Batch, Seq_Len, Num_Attrs] -> [Batch, Seq_Len] -> spłaszczamy do [Batch*Seq_Len]
            attr_targets = org_data[:, :, index].reshape(-1)

            # Obliczamy stratę
            loss = self.ce_loss(attr_logits_flat, attr_targets)

            # Dodajemy do sumy z uwzględnieniem wagi atrybutu
            if attr_name not in self.attr_weights:
                raise ValueError(
                    f"Attribute '{attr_name}' not found in attribute weights dictionary. "
                    f"Please provide a weight for this attribute."
                )
            total_recon_loss += loss * self.attr_weights[attr_name]

        # 2. Obliczanie KL Divergence Loss (Wzór analityczny dla rozkładów Gaussa)
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Normalizacja KL przez wielkość batcha (żeby nie dominowało przy dużym batchu)
        batch_size = org_data.shape[0]
        kl_loss = kl_loss / batch_size

        annealing_beta = self._get_annealing_beta(global_step)

        total_loss = total_recon_loss + (annealing_beta * kl_loss)
        name = "train" if train else "val"

        return total_loss, {
            f"{name}_recon_loss": total_recon_loss,
            f"{name}_kl_div": kl_loss,
            f"{name}_annealing_beta": annealing_beta,
        }
