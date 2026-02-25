import torch
import logging as lg

from torch.nn import functional as F
from src.models.decoder_wrappers.base_wrapper import BaseWrapper
from src.models.sequence_decoder import BaseDecoder
from src.models.utils.patch_models import PatchEmbeddings, PatchOutputHead

logger = lg.getLogger(__name__)


class PatchWrapper(BaseWrapper):
    def __init__(
        self, decoder: BaseDecoder, input_dropout_rate: float, token_embedding_size: int
    ):
        super().__init__(decoder=decoder, input_dropout_rate=input_dropout_rate)
        self._token_embedding_size = token_embedding_size
        self._patch_embedding = PatchEmbeddings(
            vocab_size=self._decoder.vocab_size,
            hidden_size=self._decoder.hidden_size,
            patch_size=self._decoder.patch_size,
            token_embedding_size=self._token_embedding_size,
        )
        self._output_proj = PatchOutputHead(
            vocab_size=self._decoder.vocab_size,
            hidden_size=self._decoder.hidden_size,
            patch_size=self._decoder.patch_size,
        )

    def forward(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor],
        *,
        latent_vector: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        input_sequence, class_idx = batch_data

        inputs_embeds = self._patch_embedding(input_sequence)
        inputs_embeds = self.input_dropout(inputs_embeds, input_sequence)
        model_state = self._decoder(
            input_embedding=inputs_embeds,
            latent_vector=latent_vector,
            class_idx=class_idx,
            use_cache=use_cache,
        )
        return self._output_proj(model_state)  # [B, seq_len, vocab_size]

    def input_dropout(
        self, embeddings: torch.Tensor, input_sequence: torch.Tensor
    ) -> torch.Tensor:
        if self.training and self._input_dropout_rate > 0.0:
            keep_prob = 1.0 - self._input_dropout_rate
            random_mask = torch.bernoulli(
                torch.full(
                    (input_sequence.shape[0], input_sequence.shape[1], 1),
                    keep_prob,
                    device=input_sequence.device,
                )
            )
            first_tokens = input_sequence[:, :, 0]
            is_protected = torch.isin(first_tokens, self._decoder.special_ids)
            protected_mask = is_protected.unsqueeze(-1).float()

            final_mask = torch.max(random_mask, protected_mask)
            embeddings = embeddings * final_mask
        return embeddings

    def generate(
        self,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        logger.info(
            f"Starting generation with decoder: {self._decoder.name} with Patch Wrapper"
        )
        generated_sequence = [[self._decoder.bos_token] * self.patch_size]
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
            )
            next_token_logits = logits[
                0, -1, :, :
            ]  # taking last ones (logits for next patch)
            next_patch = self.sample_next_item(
                next_token_logits,
                strict_sampling=strict_sampling,
                temperature=temperature,
            )
            generated_sequence.append(next_patch)
            if (
                next_patch == [self._decoder.eos_token] * self.patch_size
                or len(generated_sequence) >= self._decoder.sequence_length
            ):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self._decoder.clean_cache()
        return generated_sequence[1:]

    def sample_next_item(
        self, logits: torch.Tensor, strict_sampling: bool, temperature: float
    ) -> list[int]:
        """
        Sample the next patch based on the logits output from the model.
        If strict_sampling is True, it will take the argmax of the logits to get the most likely next patch.
        If strict_sampling is False, it will apply temperature scaling to the logits,
        convert them to probabilities using softmax, and then sample from the resulting distribution.

        :param logits: Tensor of shape [Patch_Size, Vocab_Size] containing the logits for the next patch tokens.
        :param strict_sampling: If True, use argmax to select the next patch tokens. If False, sample from the distribution defined by the logits.
        :param temperature: Temperature value for scaling the logits when strict_sampling is False. Higher values will make the distribution more uniform, while lower values will make it more peaked.
        :return: List of integers representing the next patch tokens.
        """

        if logits.dim() != 2:
            raise ValueError(
                f"Expected logits to have shape [Patch_Size, Vocab_Size], but got shape: {logits.shape}"
            )

        if strict_sampling:
            tokens = logits.argmax(dim=-1)
            return tokens.tolist()
        else:
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_patch_indices = torch.multinomial(probs, num_samples=1)
            return next_patch_indices.squeeze(-1).tolist()

    @staticmethod
    def reconstruct(
        reconstruction: torch.Tensor, x_target: torch.Tensor
    ) -> tuple[float, dict]:
        token_reconstruction = reconstruction.argmax(dim=-1)
        correctness_reconstruction = (
            token_reconstruction == x_target
        ).float().mean().item() * 100
        return correctness_reconstruction, {}

    @property
    def patch_size(self) -> int:
        return self._decoder.patch_size
