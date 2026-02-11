import torch

from src.loss.base_loss import BaseLoss


class ReconstructionLoss(BaseLoss):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
    ):
        super().__init__(vocab_size, pad_token_id)

    def __call__(
        self,
        org_data: torch.Tensor,
        reconstruction: torch.Tensor,
        global_step: int,
        train: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon_loss = torch.nn.functional.cross_entropy(
            reconstruction.view(
                -1, self._vocab_size
            ),  # Reshape to (batch*sequence_length, vocab_size)
            org_data.view(-1),  # Reshape to (batch*sequence_length)
            ignore_index=self._pad_token_id,
            reduction="mean",
        )

        return recon_loss, {}
