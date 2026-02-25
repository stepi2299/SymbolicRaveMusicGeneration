import torch
from torch import nn
import logging as lg

from torch.nn import functional as F
from src.models.decoder_wrappers.base_wrapper import BaseWrapper
from src.models.sequence_decoder import BaseDecoder


logger = lg.getLogger(__name__)


class SimpleWrapper(BaseWrapper):
    def __init__(self, decoder: BaseDecoder, input_dropout_rate: float):
        super().__init__(decoder=decoder, input_dropout_rate=input_dropout_rate)
        self.token_embeddings = nn.Embedding(
            self._decoder.vocab_size, self._decoder.hidden_size
        )
        self.output_proj = nn.Linear(
            self._decoder.hidden_size, self._decoder.vocab_size
        )

    def forward(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor],
        *,
        latent_vector: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        input_sequence, class_idx = batch_data
        token_emb = self.token_embeddings(input_sequence)
        token_emb = self.input_dropout(token_emb, input_sequence)
        model_state = self._decoder(
            input_embedding=token_emb,
            latent_vector=latent_vector,
            class_idx=class_idx,
            use_cache=use_cache,
        )
        return self.output_proj(model_state)  # [B, seq_len, vocab_size]

    def input_dropout(
        self, embeddings: torch.Tensor, input_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply dropout to the input sequence.
        Args:
            embeddings: [batch_size, sequence_length, embedding_size]
            input_sequence: [batch_size, sequence_length]
        Returns:
            dropped_out_sequence: [batch_size, sequence_length]
        """
        if self.training and self._input_dropout_rate > 0.0:
            keep_prob = 1.0 - self._input_dropout_rate
            random_mask = torch.bernoulli(
                torch.full(
                    input_sequence.shape, keep_prob, device=input_sequence.device
                )
            )
            is_protected = torch.isin(input_sequence, self._decoder.special_ids)
            final_mask = torch.max(random_mask, is_protected.float())
            embeddings = embeddings * final_mask.unsqueeze(-1)
        return embeddings

    def generate(
        self,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[int]:
        logger.debug(f"Starting generation with decoder: {self._decoder.name}")
        generated_sequence = [self._decoder.bos_token]
        self._decoder.clean_cache()
        generation_device = (
            latent_vector.get_device()
            if latent_vector is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        for _ in range(self._decoder.sequence_length):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(generation_device)
            )
            logits = self(
                [input_seq, class_idx],
                latent_vector=latent_vector,
                use_cache=use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence
            next_token_logits = logits[
                0, -1, :
            ]  # taking last one (logits for next token)

            next_token_idx = self.sample_next_item(
                next_token_logits,
                strict_sampling=strict_sampling,
                temperature=temperature,
            )
            generated_sequence.append(next_token_idx)

            if (
                next_token_idx == self._decoder.eos_token
                or len(generated_sequence) >= self._decoder.sequence_length
            ):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self._decoder.clean_cache()
        return generated_sequence[1:]

    def sample_next_item(
        self, logits: torch.Tensor, strict_sampling: bool, temperature: float
    ) -> int:
        if strict_sampling:
            return logits.argmax().item()
        probs = F.softmax(logits / temperature, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1).item()
        return next_token_idx

    @staticmethod
    def reconstruct(
        reconstruction: torch.Tensor, x_target: torch.Tensor
    ) -> tuple[float, dict]:
        token_reconstruction = reconstruction.argmax(dim=-1)
        correctness_reconstruction = (
            token_reconstruction == x_target
        ).float().mean().item() * 100
        return correctness_reconstruction, {}
