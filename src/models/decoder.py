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
        num_decoder_layers: int,
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
        self.positional_embeddings = nn.Embedding(
            self.sequence_length, self.hidden_size
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size, nhead=self.num_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(
            layer, num_layers=num_decoder_layers
        )

        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        recv_in_music_form: bool = False,
    ) -> torch.Tensor:
        # Getting current sequence length (for generation when len is growing)
        batch_size, current_sequence_len = input_sequence.shape

        position_idx = torch.arange(
            current_sequence_len, dtype=torch.long, device=input_sequence.device
        )
        position_idx = position_idx.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.positional_embeddings(position_idx)  # [1, seq_len, hidden]

        latent_vector_projection = self.latent_proj(
            latent_vector
        )  # [B, latent_dim] -> [B, embedding_size]
        latent_vector_projection = latent_vector_projection.unsqueeze(1).repeat(
            1, current_sequence_len, 1
        )  # [batch_size, seq_len, embedding_size]

        token_emb = self.token_embeddings(input_sequence)

        decoder_input = token_emb + latent_vector_projection + pos_emb
        decoder_input = decoder_input.permute(1, 0, 2)

        mask = nn.Transformer.generate_square_subsequent_mask(current_sequence_len).to(
            latent_vector.device
        )
        # TODO dac maske odpowiednia gdy token jest 0!!! (mozliwe ze to wazne tez w encoderze) THINK

        h = self.transformer_decoder(
            decoder_input,
            memory=decoder_input,
            tgt_mask=mask,
        ).transpose(0, 1)
        output_proj = self.output_proj(h)  # [B, seq_len, vocab_size]
        if recv_in_music_form:
            output_proj = self.transform_to_music_form(output_proj)  # [B, seq_len]
        return output_proj


def decoder_mapper(decoder_name: str) -> Type[Decoder]:
    if decoder_name == "gpt_decoder":
        return GptDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
