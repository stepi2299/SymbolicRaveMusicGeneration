import torch
from torch import nn
from torch.nn import functional as F
import logging as lg

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.adapter import MusicAdapter

logger = lg.getLogger(__name__)


class VAE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        adapter: MusicAdapter,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._adapter = adapter
        self._validate_if_latent_dimensions_match()
        self._orig_sample = None
        self._use_cache = use_cache_generation
        self._strict_sampling = strict_sampling

    def _validate_if_latent_dimensions_match(self) -> None:
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                "Encoder and Decoder latent dimensions must match, "
                f"got {self.encoder.latent_dim} for encoder and {self.decoder.latent_dim} for decoder"
            )

    def forward(
        self, input_music: torch.Tensor, recv_in_music_form: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._orig_sample = input_music
        mu, log_variance = self.encoder(input_music)
        latent_vector = self.reparameterize(mu, log_variance)
        tensor_reconstruction = self.decoder(
            latent_vector,
            input_music,
            recv_in_music_form=recv_in_music_form,
            use_cache=False,
        )
        return tensor_reconstruction, mu, log_variance

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate(self, latent_vector: torch.Tensor) -> list[list[int]]:
        self.decoder.eval()
        if latent_vector.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent vector must have shape (batch_size, {self.latent_dim}), got {latent_vector.shape}"
            )
        songs = []
        for i, single_condition_latent in enumerate(latent_vector):
            try:
                if self._adapter.song_dim == 1:
                    song = self._generate_single_song(
                        single_condition_latent.unsqueeze(0)
                    )
                elif self._adapter.song_dim == 2:
                    song = self._generate_song_for_2_dim(
                        single_condition_latent.unsqueeze(0)
                    )
                else:
                    raise ValueError(
                        f"Unsupported song dimension {self._adapter.song_dim}"
                    )
                songs.append(song)
            except Exception as e:
                logger.error(
                    f"Error generating song for latent vector for index: {i}: {e}"
                )
        return songs

    @staticmethod
    def sample_next_token(logits, temperature=1.0, strict: bool = False) -> int:
        if strict:
            return logits.argmax().item()
        probs = F.softmax(logits / temperature, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1).item()
        return next_token_idx

    def _generate_single_song(self, latent_vector: torch.Tensor) -> list[int]:
        generated_sequence = [self._adapter.bos_token_id]
        self.decoder.clean_cache()
        for _ in range(self.decoder.sequence_length):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(latent_vector.device)
            )
            logits = self.decoder(
                latent_vector,
                input_seq,
                recv_in_music_form=False,
                use_cache=self._use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence
            next_token_logits = logits[
                0, -1, :
            ]  # taking last one (logits for next token)

            next_token_idx = self.sample_next_token(
                next_token_logits, strict=self._strict_sampling
            )
            generated_sequence.append(next_token_idx)

            if (
                next_token_idx == self._adapter.eos_token_id
                or len(generated_sequence) >= self.decoder.sequence_length
            ):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self.decoder.clean_cache()
        return generated_sequence[1:]

    def _generate_song_for_2_dim(self, latent_vector: torch.Tensor) -> list[list[int]]:
        logger.info("Generating song for 2D representation")
        generated_sequence = [
            [self._adapter.bos_token_id] * self._adapter.patch_size
        ]
        self.decoder.clean_cache()
        for i in range(self.decoder.sequence_length - 1):
            logger.debug(f"Generated sequence: {generated_sequence}")
            try:
                input_seq = (
                    torch.tensor(generated_sequence)
                    .unsqueeze(0)
                    .to(latent_vector.device)
                )
            except Exception as e:
                logger.error(f"Error creating input sequence tensor at step {i}: {e}")
                raise
            logger.debug(f"Input Sequence - Tensor: {input_seq.shape}")
            dict_logits = self.decoder(
                latent_vector,
                input_seq,
                recv_in_music_form=False,
                use_cache=self._use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence
            logger.debug(
                f"Recv Logits with keys: {dict_logits.keys()} with shapes: {[v.shape for v in dict_logits.values()]}"
            )
            next_patch = self.sample_next_patch(dict_logits)
            # logger.debug(f"Next Patch: {next_patch}, old patch: {self._orig_sample[0][i+1]}")
            generated_sequence.append(next_patch)
            logger.debug(
                f"Current Generated Sequence Length: {len(generated_sequence)}"
            )
            if len(
                generated_sequence
            ) >= self.decoder.sequence_length or self._patch_is_eos_or_pad(next_patch):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self.decoder.clean_cache()
        return generated_sequence[1:]

    def load_pretrained(self) -> None:
        self.encoder.load_pretrained()

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

    def _eos_patch_equal(self, next_patch: list[int]) -> bool:
        return next_patch == [self._adapter.eos_token_id] * self._adapter.patch_size

    def _patch_is_eos_or_pad(self, next_patch: list[int]) -> bool:
        allowed = {self._adapter.eos_token_id, self._adapter.pad_token_id}
        return set(next_patch).issubset(allowed)

    @staticmethod
    def sample_next_patch(
        logits: dict[str, torch.Tensor], strict: bool = True
    ) -> list[int]:
        """
        Sample next patch tokens from the logits for each attribute.
        Args:
            logits: Dict of [batch_size, vocab_size] with key as attribute name (Logits for last sequence position)
            strict: If True, use argmax sampling
        Returns:
            next_patch: List of length patch_size with sampled tokens
        """
        next_patch = []
        for attr, attr_logits in logits.items():
            if strict:
                all_current_tokens = attr_logits.argmax(
                    dim=-1
                )  # [batch_size, sequence_length]
                logger.debug(
                    f"Current tokens for attribute {attr}: {all_current_tokens.shape}"
                )
                next_token = all_current_tokens[0, -1]
            else:
                probs = torch.softmax(attr_logits, dim=-1)  # [batch_size, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)[-1]
            logger.debug(f"Next token for attribute {attr}: {next_token.item()}")
            next_patch.append(next_token.item())
        return next_patch
