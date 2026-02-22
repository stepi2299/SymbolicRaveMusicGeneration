import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block.

    This module applies a conditional affine transformation to the input features.
    It takes a categorical condition (e.g., a music genre class index), embeds it,
    and maps it to two conditioning parameters: gamma (scale) and beta (shift).
    The input features are then modulated using the equation: FiLM(x) = gamma * x + beta.

    This mechanism allows the neural network to dynamically adjust its feature maps
    based on the provided condition, making it highly effective for conditional
    music generation.
    """

    def __init__(self, num_classes: int, embedding_dim: int, feature_dim: int):
        super().__init__()

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.film_generator = nn.Linear(embedding_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor of shape [batch_size, feature_dim, sequence_length]
            class_idx: Tensor containing class indices of shape [batch_size]

        Returns:
            Modulated feature tensor of the same shape as input `x`.
        """
        cond_emb = self.class_embedding(class_idx)
        film_params = self.film_generator(cond_emb)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)

        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return (gamma * x) + beta
