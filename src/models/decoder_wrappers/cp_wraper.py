import torch
import logging as lg

from src.models.decoder_wrappers.base_wrapper import BaseWrapper
from src.models.sequence_decoder import BaseDecoder
from src.models.utils.cp_word import MultiHeadOutput, CompoundWordEmbedding
from src.constants import CP_WORD_INDEXES


logger = lg.getLogger(__name__)


class CompoundWordWrapper(BaseWrapper):
    def __init__(self, decoder: BaseDecoder, input_dropout_rate: float):
        super().__init__(decoder=decoder, input_dropout_rate=input_dropout_rate)
        # 1. Input Module (Adaptive Embeddings)
        self.cp_embedding = CompoundWordEmbedding(self._decoder.hidden_size)
        self.output_proj = MultiHeadOutput(self._decoder.hidden_size)

    def forward(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor],
        *,
        latent_vector: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        input_sequence, class_idx = batch_data
        batch_size, seq_len, num_attrs = input_sequence.shape

        assert (
            num_attrs == len(CP_WORD_INDEXES)
        ), f"Liczba atrybutów w input_sequence nie zgadza się z configiem! Otrzymano {num_attrs}, oczekiwano {len(CP_WORD_INDEXES)}"

        # A. Embed Inputs
        # Zamieniamy [B, L, Attrs] -> [B, L, Hidden]
        inputs_embeds = self.cp_embedding(input_sequence)
        inputs_embeds = self.input_dropout(
            inputs_embeds, input_sequence, CP_WORD_INDEXES["family"]
        )
        model_state = self._decoder(
            input_embedding=inputs_embeds,
            latent_vector=latent_vector,
            class_idx=class_idx,
            use_cache=use_cache,
        )
        return self.output_proj(model_state)  # [B, seq_len, vocab_size]

    def input_dropout(
        self,
        embeddings: torch.Tensor,
        input_sequence: torch.Tensor,
        family_attr_index: int = 0,
    ) -> torch.Tensor:
        """
        Dropout only on non-protected tokens based on 'family' attribute.

        Args:
            embeddings: [Batch, Seq_Len, Hidden_Size] (Embedded input sequence)
            input_sequence: [Batch, Seq_Len, Num_Attributes] (Original input sequence with attributes)
            family_attr_index: Index of the 'family' attribute in input_sequence
        Returns:
            embeddings: [Batch, Seq_Len, Hidden_Size]
        """
        if self.training and self._input_dropout_rate > 0.0:
            keep_prob = 1.0 - self._input_dropout_rate
            random_mask = torch.bernoulli(
                torch.full(
                    (embeddings.shape[0], embeddings.shape[1]),
                    keep_prob,
                    device=embeddings.device,
                )
            )

            family_tokens = input_sequence[:, :, family_attr_index]
            is_protected = torch.isin(
                family_tokens, torch.tensor(self.special_ids, device=embeddings.device)
            )
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
    ) -> list[list[int]]:
        logger.info(
            f"Starting generation with decoder: {self._decoder.name} in CP Wrapper"
        )
        generated_sequence = [[self._decoder.bos_token] * self.patch_size]
        self._decoder.clean_cache()
        generation_device = (
            latent_vector.get_device()
            if latent_vector is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        for i in range(self._decoder.sequence_length - 1):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(generation_device)
            )
            dict_logits = self(
                [input_seq, class_idx],
                latent_vector=latent_vector,
                use_cache=use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence

            next_patch = self.sample_next_item(
                dict_logits, strict_sampling=strict_sampling, temperature=temperature
            )
            generated_sequence.append(next_patch)
            if (
                len(generated_sequence) >= self._decoder.sequence_length
                or next_patch == [self._decoder.eos_token] * self.patch_size
            ):
                break
        self._decoder.clean_cache()
        return generated_sequence[1:]

    def sample_next_item(
        self, logits: dict[str, torch.Tensor], strict_sampling: bool, temperature: float
    ) -> list[int]:
        """
        Sample next patch tokens from the logits for each attribute.
        Args:
            logits: Dictionary of attribute names to their corresponding logits tensors of shape [batch_size, vocab_size]
            strict_sampling: If True, take the argmax token for each attribute. If False, sample from the distribution.
            temperature: Temperature for sampling (only used if strict_sampling is False)
        Returns:
            next_patch: List of length patch_size with sampled tokens
        """
        next_patch = []
        for attr, attr_logits in logits.items():
            if strict_sampling:
                all_current_tokens = attr_logits.argmax(
                    dim=-1
                )  # [batch_size, sequence_length]
                next_token = all_current_tokens[0, -1]
            else:
                probs = torch.softmax(attr_logits, dim=-1)  # [batch_size, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)[-1]
            logger.debug(f"Next token for attribute {attr}: {next_token.item()}")
            next_patch.append(next_token.item())
        return next_patch

    @staticmethod
    def reconstruct(
        reconstruction: dict[str, torch.Tensor], x_target: torch.Tensor
    ) -> tuple[float, dict]:
        total_accuracy = 0.0
        metadata = {}
        for attr_name, logits in reconstruction.items():
            attr_index = CP_WORD_INDEXES[attr_name]
            token_reconstruction = logits.argmax(dim=-1)
            correctness_reconstruction = (
                token_reconstruction == x_target[:, :, attr_index]
            ).float().mean().item() * 100
            total_accuracy += correctness_reconstruction
            metadata[f"reconstruction_success_{attr_name}"] = correctness_reconstruction
        avg_accuracy = total_accuracy / len(reconstruction)
        return avg_accuracy, metadata

    @property
    def patch_size(self) -> int:
        return self._decoder.patch_size
