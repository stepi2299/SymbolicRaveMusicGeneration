from src.loss.base_loss import BaseLoss
from src.loss.elbo_loss import ElboLoss
from src.loss.cp_elbo_loss import CpElboLoss
from src.loss.reconstruction_loss import ReconstructionLoss


def get_loss_function(
    loss_name: str, vocab_size: int, pad_token_id: int, **kwargs
) -> BaseLoss:
    loss_mapper: dict = {
        "elbo": ElboLoss,
        "cp_elbo": CpElboLoss,
        "reconstruction": ReconstructionLoss,
    }
    if loss_name not in loss_mapper:
        raise ValueError(
            f"Loss function '{loss_name}' is not recognized. Available losses: {list(loss_mapper.keys())}"
        )
    return loss_mapper[loss_name](
        vocab_size=vocab_size, pad_token_id=pad_token_id, **kwargs
    )


__all__ = [
    "BaseLoss",
    "get_loss_function",
    "ElboLoss",
    "CpElboLoss",
    "ReconstructionLoss",
]
