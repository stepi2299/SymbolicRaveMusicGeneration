import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type


class Decoder(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        name: str,
        sequence_length: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size

    @abstractmethod
    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        recv_in_music_form: bool = False,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def transform_to_music_form(recv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Transform the received tokens to the music form.
        Args:
            recv_tokens: [batch_size, sequence_length, patch_size, vocab_size]
        Returns:
            music_form: [batch_size, sequence_length, patch_size]
        """

        return recv_tokens.argmax(dim=-1)


class GptDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
    ) -> None:
        super(GptDecoder, self).__init__(
            latent_dim=latent_dim,
            name="gpt",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
        )
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_size)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size, nhead=self.num_heads
        )
        self._sequence_of_patches = False

        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        recv_in_music_form: bool = False,
    ) -> torch.Tensor:
        # Getting current sequence length (for generation when len is growing)
        current_sequence_len = input_sequence.size(1)
        latent_vector_projection = self.latent_proj(
            latent_vector
        )  # [B, latent_dim] -> [B, embedding_size]
        latent_vector_projection = latent_vector_projection.unsqueeze(1).repeat(
            1, current_sequence_len * self.patch_size, 1
        )  # [batch_size, seq_len, embedding_size]

        if input_sequence.dim() == 3:
            self._sequence_of_patches = True
            input_sequence = input_sequence.reshape(
                -1, self.patch_size * self.sequence_length
            )
        token_emb = self.token_embeddings(input_sequence)
        decoder_input = token_emb + latent_vector_projection

        mask = nn.Transformer.generate_square_subsequent_mask(
            current_sequence_len * self.patch_size
        ).to(latent_vector.device)

        h = self.transformer_layer(
            decoder_input.transpose(0, 1),
            memory=decoder_input.transpose(0, 1),
            tgt_mask=mask,
        ).transpose(0, 1)
        output_proj = self.output_proj(h)
        if self._sequence_of_patches:
            output_proj = output_proj.view(
                latent_vector.size(0),
                self.sequence_length,
                self.patch_size,
                self.vocab_size,
            )
            self._sequence_of_patches = False
        if recv_in_music_form:
            output_proj = self.transform_to_music_form(output_proj)
        return output_proj


def decoder_mapper(decoder_name: str) -> Type[Decoder]:
    if decoder_name == "gpt_decoder":
        return GptDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
