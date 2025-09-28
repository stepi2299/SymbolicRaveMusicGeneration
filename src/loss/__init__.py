from src.loss.base_vae_loss import BaseVaeLoss
from src.loss.elbo_loss import ElboLoss


def get_loss_function(loss_name: str, vocab_size: int) -> BaseVaeLoss:
    loss_mapper = {
        "elbo": ElboLoss,
        # Future loss functions can be added here
    }
    if loss_name not in loss_mapper:
        raise ValueError(
            f"Loss function '{loss_name}' is not recognized. Available losses: {list(loss_mapper.keys())}"
        )
    return loss_mapper[loss_name](vocab_size)


__all__ = ["BaseVaeLoss", "get_loss_function"]
