import torch
from torch import nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) module.

    This module replaces the standard learnable scale and shift parameters
    of LayerNorm with parameters dynamically generated from a conditioning vector.

    Attributes:
        norm (nn.LayerNorm): Standard layer normalization without its own learnable
                             elementwise affine parameters (gamma and beta).
        emb2ada (nn.Linear): Linear projection layer mapping the conditioning vector
                             to the required scale and shift factors.
    """

    def __init__(self, hidden_size: int, *, eps: float, cond_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.emb2ada = nn.Linear(cond_size, 2 * hidden_size)

        """
        Zero-initialization of the linear layer's weights and biases.
        This ensures that at the beginning of training, the model behaves exactly
        like a standard, unconditioned Transformer. The scale multiplier starts at 0
        (which we add to 1 later) and shift starts at 0.
        """
        nn.init.zeros_(self.emb2ada.weight)
        nn.init.zeros_(self.emb2ada.bias)

    def forward(self, input_activation: torch.Tensor, conditioning_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaLN module.

        Args:
            input_activation (torch.Tensor): Input activation tensor of shape [batch_size, seq_len, hidden_size].
            conditioning_vector (torch.Tensor): Conditioning vector (e.g., class embedding)
                                 of shape [batch_size, cond_size].

        Returns:
            torch.Tensor: The normalized tensor modulated by the conditioning vector.
        """
        ada_params = self.emb2ada(conditioning_vector)
        scale, shift = ada_params.chunk(2, dim=-1)

        return self.norm(input_activation) * (1 + scale) + shift
