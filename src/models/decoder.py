import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type
import logging as lg
from transformers import (
    TransfoXLConfig,
    TransfoXLModel,
    GPTNeoXConfig,
    GPTNeoXModel,
    MambaConfig,
    MambaModel,
    MistralConfig,
    MistralModel,
)
from src.models.linear_attention import LinearTransformerBlock
from src.models.cp_word import MultiHeadOutput, CompoundWordEmbedding
from src.constants import CP_WORD_INDEXES, Conditioning
from src.models.patch_models import PatchEmbeddings, PatchOutputHead
from src.utils import sample_next_cp_word, sample_next_token, sample_next_patch
from src.models.custom_rope import CustomGPTNeoXModel

logger = lg.getLogger(__name__)


class Decoder(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        name: str,
        sequence_length: int,
        patch_size: int,
        bos_token: int,
        eos_token: int,
        pad_token: int,
        hidden_size: int,
        conditioning: str,
        dropout: float = 0.1,
        num_decoder_layers: int = 1,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        self.intermediate_ff_size = 4 * hidden_size
        self._input_dropout = input_dropout
        self._cache_params = None
        special_token_ids = [self.bos_token, self.eos_token, self.pad_token]
        self.register_buffer(
            "special_ids", torch.tensor(special_token_ids, dtype=torch.long)
        )
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.latent_proj = nn.Linear(latent_dim, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)
        self._concat_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self._conditioning = Conditioning(conditioning)
        logger.info(
            f"Initialize Decoder: {self.__class__.__name__} with conditioning: {self._conditioning}"
        )

    def _condition_input(
        self, token_embedding: torch.Tensor, latent_projection: torch.Tensor
    ) -> torch.Tensor:
        match self._conditioning:
            case Conditioning.ADD:
                return token_embedding + latent_projection
            case Conditioning.CONCATENATION:
                concat_input = torch.cat(
                    [token_embedding, latent_projection], dim=-1
                )  # [B, Seq_Len, Hidden*2]
                return self._concat_proj(concat_input)  # [B, Seq_Len, Hidden]
            case _:
                raise ValueError(
                    f"Unsupported conditioning method: {self._conditioning}"
                )

    def _prepare_input(
        self, input_sequence: torch.Tensor, latent_vector: torch.Tensor
    ) -> torch.Tensor:
        batch_size, current_sequence_len = input_sequence.shape
        token_emb = self.token_embeddings(input_sequence)
        token_emb = self.input_dropout(token_emb, input_sequence)

        # [B, Latent] -> [B, Hidden]
        latent_proj = self.latent_proj(latent_vector)

        # [B, 1, Hidden] -> [B, Seq_Len, Hidden]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        return self._condition_input(
            token_embedding=token_emb, latent_projection=latent_proj_expanded
        )

    @abstractmethod
    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
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

    @abstractmethod
    def get_layers_to_track(self):
        pass

    def clean_cache(self):
        self._cache_params = None

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
        if self.training and self._input_dropout > 0.0:
            keep_prob = 1.0 - self._input_dropout
            random_mask = torch.bernoulli(
                torch.full(
                    input_sequence.shape, keep_prob, device=input_sequence.device
                )
            )
            is_protected = torch.isin(input_sequence, self.special_ids)
            final_mask = torch.max(random_mask, is_protected.float())
            embeddings = embeddings * final_mask.unsqueeze(-1)
        return embeddings

    @staticmethod
    def reconstruct(
        reconstruction: torch.Tensor, x_target: torch.Tensor
    ) -> tuple[float, dict]:
        token_reconstruction = reconstruction.argmax(dim=-1)
        correctness_reconstruction = (
            token_reconstruction == x_target
        ).float().mean().item() * 100
        return correctness_reconstruction, {}

    def generate(
        self,
        latent_vector: torch.Tensor,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[int] | list[list[int]]:
        logger.debug(f"Starting generation with decoder: {self.name}")
        generated_sequence = [self.bos_token]
        self.clean_cache()
        for _ in range(self.sequence_length):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(latent_vector.device)
            )
            logits = self(
                latent_vector,
                input_seq,
                recv_in_music_form=False,
                use_cache=use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence
            next_token_logits = logits[
                0, -1, :
            ]  # taking last one (logits for next token)

            next_token_idx = sample_next_token(
                next_token_logits, strict=strict_sampling, temperature=temperature
            )
            generated_sequence.append(next_token_idx)

            if (
                next_token_idx == self.eos_token
                or len(generated_sequence) >= self.sequence_length
            ):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self.clean_cache()
        return generated_sequence[1:]


class GptDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        dropout: float,
        input_dropout: float,
        conditioning: str,
    ) -> None:
        super(GptDecoder, self).__init__(
            latent_dim=latent_dim,
            name="gpt",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=dropout,
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )

        self.positional_embeddings = nn.Embedding(
            self.sequence_length, self.hidden_size
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.intermediate_ff_size,
            dropout=self.dropout,
            norm_first=True,
        )
        self.model = nn.TransformerDecoder(layer, num_layers=self.num_decoder_layers)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Getting current sequence length (for generation when len is growing)
        batch_size, current_sequence_len = input_sequence.shape

        position_idx = torch.arange(
            current_sequence_len, dtype=torch.long, device=input_sequence.device
        )
        position_idx = position_idx.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.positional_embeddings(position_idx)  # [1, seq_len, hidden]

        decoder_input = self._prepare_input(input_sequence, latent_vector)
        decoder_input = decoder_input + pos_emb
        decoder_input = decoder_input.permute(1, 0, 2)

        mask = nn.Transformer.generate_square_subsequent_mask(current_sequence_len).to(
            latent_vector.device
        )

        h = self.model(
            decoder_input,
            memory=decoder_input,
            tgt_mask=mask,
        ).transpose(0, 1)
        output_proj = self.output_proj(h)  # [B, seq_len, vocab_size]
        if recv_in_music_form:
            output_proj = self.transform_to_music_form(output_proj)  # [B, seq_len]
        return output_proj

    def get_layers_to_track(self):
        return self.model.layers


class TransformerXL(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        mem_len: int,
        dropout: float,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        attention_dropout: float,
        input_dropout: float,
        conditioning: str,
    ) -> None:
        super(TransformerXL, self).__init__(
            latent_dim=latent_dim,
            name="transformer_xl",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )

        config = TransfoXLConfig(
            d_model=self.hidden_size,
            n_head=self.num_heads,
            n_layer=num_decoder_layers,
            d_inner=self.intermediate_ff_size,
            vocab_size=self.vocab_size,
            mem_len=mem_len,  # Length of the memory
            dropout=self.dropout,
            dropatt=attention_dropout,
            # Important! Disable adaptive input and output embeddings
            adaptive=False,
            div_val=1,  # No division of embedding size,
            pre_lnorm=True,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
        )

        self.model = TransfoXLModel(config)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(input_sequence, latent_vector)

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds, mems=self._cache_params
        )
        last_hidden_state = outputs.last_hidden_state  # [B, Seq_Len, Hidden]
        self._cache_params = outputs.mems

        logits = self.output_proj(last_hidden_state)  # [B, Seq_Len, Vocab]
        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)
        return logits

    def get_layers_to_track(self):
        return self.model.layers


class TransformerRope(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        rotary_pct: float,
        rotary_emb_base: int,
        conditioning: str,
        custom: bool,
        use_cross_attention: bool,
        input_dropout: float = 0.0,
    ) -> None:
        super(TransformerRope, self).__init__(
            latent_dim=latent_dim,
            name="transformer_rope",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )

        config = GPTNeoXConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_decoder_layers,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=vocab_size,
            classifier_dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=sequence_length,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )
        self.custom = custom
        if custom:
            self.model = CustomGPTNeoXModel(
                config, use_cross_attention=use_cross_attention
            )
        else:
            self.model = GPTNeoXModel(config)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,  # set True during inference
    ) -> torch.Tensor:
        batch_size, current_sequence_len = input_sequence.shape
        token_emb = self.token_embeddings(input_sequence)
        token_emb = self.input_dropout(token_emb, input_sequence)

        # [B, Latent] -> [B, Hidden]
        latent_proj = self.latent_proj(latent_vector)

        # [B, 1, Hidden] -> [B, Seq_Len, Hidden]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        if self.custom:
            outputs = self.model(
                inputs_embeds=token_emb,
                encoder_hidden_states=latent_proj_expanded,
                past_key_values=self._cache_params,
                return_dict=True,
                use_cache=use_cache,
            )
        else:
            decoder_input_embeds = self._condition_input(
                token_embedding=token_emb, latent_projection=latent_proj_expanded
            )
            outputs = self.model(
                inputs_embeds=decoder_input_embeds,
                past_key_values=self._cache_params,
                return_dict=True,
                use_cache=use_cache,
            )

        hidden_states = outputs.last_hidden_state

        logits = self.output_proj(hidden_states)  # [B, Seq_Len, Vocab]
        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)
        if use_cache:
            self._cache_params = outputs.past_key_values
        return logits

    def get_layers_to_track(self):
        return self.model.layers


class MistralDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        attention_dropout: float,
        rope_theta: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        sliding_window: int,
        num_key_value_heads: int,
        input_dropout: float,
        conditioning: str,
    ) -> None:
        super(MistralDecoder, self).__init__(
            latent_dim=latent_dim,
            name="mistral_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            hidden_size=hidden_size,
            dropout=attention_dropout,
            num_decoder_layers=num_decoder_layers,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )
        config = MistralConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_decoder_layers,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=vocab_size,
            max_position_embeddings=self.sequence_length,
            attention_dropout=attention_dropout,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
        )
        self.model = MistralModel(config)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,  # set True during inference
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(input_sequence, latent_vector)

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=self._cache_params,
            return_dict=True,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.output_proj(hidden_states)
        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)
        if use_cache:
            self._cache_params = outputs.past_key_values
        return logits

    def get_layers_to_track(self):
        return self.model.layers


class MambaDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        state_size: int,
        num_decoder_layers: int,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        expand: int,
        conv_kernel: int,
        input_dropout: float,
        conditioning: str,
    ) -> None:
        super(MambaDecoder, self).__init__(
            latent_dim=latent_dim,
            name="mamba_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )
        config = MambaConfig(
            hidden_size=self.hidden_size,
            state_size=state_size,
            num_hidden_layers=self.num_decoder_layers,
            vocab_size=vocab_size,
            expand=expand,
            conv_kernel=conv_kernel,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )
        self.model = MambaModel(config)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        decoder_inputs_embeds = self._prepare_input(input_sequence, latent_vector)

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            cache_params=self._cache_params,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.output_proj(hidden_states)

        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)
        if use_cache:
            self._cache_params = outputs.cache_params
        return logits

    def get_layers_to_track(self):
        return self.model.layers


# class LinearAttentionDecoder(Decoder):
#     def __init__(
#         self,
#         latent_dim: int,
#         vocab_size: int,
#         hidden_size: int,
#         sequence_length: int,
#         num_decoder_layers: int,
#         patch_size: int,
#         bos_token: int,
#         eos_token: int,
#         pad_token: int,
#         dropout: float,
#         input_dropout: float,
#     ) -> None:
#         super().__init__(
#             latent_dim=latent_dim,
#             name="mamba_decoder",
#             vocab_size=vocab_size,
#             sequence_length=sequence_length,
#             patch_size=patch_size,
#             bos_token=bos_token,
#             eos_token=eos_token,
#             pad_token=pad_token,
#             dropout=dropout,
#             num_decoder_layers=num_decoder_layers,
#             hidden_size=hidden_size,
#             input_dropout=input_dropout,
#         )
#
#         self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
#         self.position_embeddings = nn.Embedding(sequence_length, hidden_size)
#
#         self.latent_proj = nn.Linear(latent_dim, hidden_size)
#
#         # Stos warstw liniowych
#         self.layers = nn.ModuleList(
#             [
#                 LinearTransformerBlock(
#                     hidden_size, self.num_heads, self.intermediate_ff_size, dropout
#                 )
#                 for _ in range(num_decoder_layers)
#             ]
#         )
#
#         self.ln_f = nn.LayerNorm(hidden_size)
#         self.output_proj = nn.Linear(hidden_size, vocab_size)
#
#     def forward(
#         self,
#         latent_vector: torch.Tensor,
#         input_sequence: torch.Tensor,
#         *,
#         recv_in_music_form: bool = False,
#         use_cache: bool = False,
#     ) -> torch.Tensor:
#         batch_size, seq_len = input_sequence.shape
#
#         # Embeddings
#         token_emb = self.token_embeddings(input_sequence)
#         token_emb = self.input_dropout(token_emb, input_sequence)
#
#         # Add position
#         if not use_cache:  # W treningu dodajemy pozycje dla całego ciągu
#             positions = torch.arange(0, seq_len, device=token_emb.device).unsqueeze(0)
#             token_emb = token_emb + self.position_embeddings(positions)
#         else:  # W inference dla pojedynczego kroku (trzeba by śledzić index, tu uproszczenie)
#             pass
#             # Conditioning
#         latent_proj = self.latent_proj(latent_vector).unsqueeze(1)
#         token_emb = token_emb + latent_proj  # Broadcasting
#
#         new_states = []
#
#         for i, layer in enumerate(self.layers):
#             layer_state = (
#                 self._cache_params[i] if self._cache_params is not None else None
#             )
#             token_emb, new_s = layer(token_emb, state=layer_state, use_cache=use_cache)
#             if use_cache:
#                 new_states.append(new_s)
#
#         token_emb = self.ln_f(token_emb)
#         logits = self.output_proj(token_emb)
#
#         if recv_in_music_form:
#             logits = self.transform_to_music_form(logits)
#
#         if use_cache:
#             self._cache_params = new_states
#
#         return logits
#
#     def get_layers_to_track(self):
#         return self.layers


class MistralCompoundDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        attention_dropout: float,
        rope_theta: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        sliding_window: int,
        num_key_value_heads: int,
        input_dropout: float,
        conditioning: str,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name="mistral_compound_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=attention_dropout,
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )

        # 1. Input Module (Adaptive Embeddings)
        self.cp_embedding = CompoundWordEmbedding(self.hidden_size)

        # 3. Transformer Backbone (Mistral)
        config = MistralConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_decoder_layers,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=vocab_size,
            max_position_embeddings=self.sequence_length,
            attention_dropout=attention_dropout,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
        )
        self.model = MistralModel(config)

        # 4. Output Module (Multi-head)
        self.output_heads = MultiHeadOutput(hidden_size)

    def forward(
        self,
        latent_vector: torch.Tensor,  # [Batch, Latent_Dim]
        input_sequence: torch.Tensor,  # [Batch, Seq_Len, Num_Attributes] <-- ZMIANA: 3D Input
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ):
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

        # B. Inject Latent Vector (Global Conditioning)
        # [B, Latent] -> [B, Hidden]
        z_proj = self.latent_proj(latent_vector).unsqueeze(1)
        self._condition_input(token_embedding=inputs_embeds, latent_projection=z_proj)

        # C. Transformer Pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=self._cache_params,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, Hidden]

        # D. Multi-head Output
        # Zwraca słownik: {'pitch': Logits, 'duration': Logits, ...}
        logits = self.output_heads(hidden_states)

        if use_cache:
            self._cache_params = outputs.past_key_values

        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)

        return logits

    def get_layers_to_track(self):
        return self.model.layers

    def transform_to_music_form(
        self, recv_tokens: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Transform the received tokens to the music form.
        Args:
            recv_tokens: Dict of [batch_size, sequence_length, vocab_size] with key as attribute name
            (patch number of keys)
        Returns:
            music_form: Tensor of [batch_size, patch_size, sequence_length]
        """
        music_form = []
        for attr, logits in recv_tokens.items():
            music_form.append(logits.argmax(dim=-1))
        music_form = torch.stack(music_form, dim=-1)
        return music_form

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

    def generate(
        self,
        latent_vector: torch.Tensor,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        logger.info(f"Starting generation with decoder: {self.name}")
        generated_sequence = [[self.bos_token] * self.patch_size]
        self.clean_cache()
        for i in range(self.sequence_length - 1):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(latent_vector.device)
            )
            dict_logits = self(
                latent_vector,
                input_seq,
                recv_in_music_form=False,
                use_cache=use_cache,
            )  # (1, seq_len, VOCAB_SIZE) - logits for each token in the sequence

            next_patch = sample_next_cp_word(dict_logits, strict=strict_sampling)
            generated_sequence.append(next_patch)
            if (
                len(generated_sequence) >= self.sequence_length
                or next_patch == [self.eos_token] * self.patch_size
            ):
                break
        self.clean_cache()
        return generated_sequence[1:]

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
        if self.training and self._input_dropout > 0.0:
            keep_prob = 1.0 - self._input_dropout
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


class MistralPatchDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        hidden_size: int,
        sequence_length: int,
        patch_size: int,
        num_decoder_layers: int,
        attention_dropout: float,
        rope_theta: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        sliding_window: int,
        num_key_value_heads: int,
        input_dropout: float,
        token_embedding_size: int,
        conditioning: str,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name="patch_mistral_decoder",
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            patch_size=patch_size,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            dropout=attention_dropout,
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            input_dropout=input_dropout,
            conditioning=conditioning,
        )

        self.patch_embedding = PatchEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            patch_size=patch_size,
            token_embedding_size=token_embedding_size,
        )

        config = MistralConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=self.num_decoder_layers,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.intermediate_ff_size,
            vocab_size=1,
            max_position_embeddings=self.sequence_length,
            attention_dropout=attention_dropout,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
        )
        self.model = MistralModel(config)

        self.patch_output_head = PatchOutputHead(
            vocab_size=self.vocab_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
        )

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, patch_size = input_sequence.shape
        assert (
            patch_size == self.patch_size
        ), f"Expected patch size {self.patch_size}, got {patch_size}"

        patch_emb = self.patch_embedding(input_sequence)
        patch_emb = self.input_dropout(patch_emb, input_sequence)

        latent_proj = self.latent_proj(latent_vector)
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, seq_len, self.hidden_size
        )
        decoder_inputs_embeds = self._condition_input(
            token_embedding=patch_emb, latent_projection=latent_proj_expanded
        )

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=self._cache_params,
            return_dict=True,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.patch_output_head(hidden_states)

        if recv_in_music_form:
            logits = self.transform_to_music_form(logits)
        if use_cache:
            self._cache_params = outputs.past_key_values
        return logits

    def get_layers_to_track(self):
        return self.model.layers

    def generate(
        self,
        latent_vector: torch.Tensor,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        logger.info(f"Starting generation with decoder: {self.name}")
        generated_sequence = [[self.bos_token] * self.patch_size]
        self.clean_cache()
        for _ in range(self.sequence_length):
            input_seq = (
                torch.tensor(generated_sequence).unsqueeze(0).to(latent_vector.device)
            )
            logits = self(
                latent_vector,
                input_seq,
                recv_in_music_form=False,
                use_cache=use_cache,
            )
            next_token_logits = logits[
                0, -1, :, :
            ]  # taking last ones (logits for next patch)
            next_patch = sample_next_patch(next_token_logits, strict=strict_sampling)
            generated_sequence.append(next_patch)
            if (
                next_patch == [self.eos_token] * self.patch_size
                or len(generated_sequence) >= self.sequence_length
            ):
                logger.debug(f"Generated sequence: {generated_sequence}")
                break
        self.clean_cache()
        return generated_sequence[1:]

    def input_dropout(
        self, embeddings: torch.Tensor, input_sequence: torch.Tensor
    ) -> torch.Tensor:
        if self.training and self._input_dropout > 0.0:
            keep_prob = 1.0 - self._input_dropout
            random_mask = torch.bernoulli(
                torch.full(
                    (input_sequence.shape[0], input_sequence.shape[1], 1),
                    keep_prob,
                    device=input_sequence.device,
                )
            )
            first_tokens = input_sequence[:, :, 0]
            is_protected = torch.isin(first_tokens, self.special_ids)
            protected_mask = is_protected.unsqueeze(-1).float()

            final_mask = torch.max(random_mask, protected_mask)
            embeddings = embeddings * final_mask
        return embeddings


def decoder_mapper(decoder_name: str) -> Type[Decoder]:
    if decoder_name == "gpt_decoder":
        return GptDecoder
    elif decoder_name == "transformer_xl":
        return TransformerXL
    elif decoder_name == "transformer_rope":
        return TransformerRope
    elif decoder_name == "mistral":
        return MistralDecoder
    elif decoder_name == "mamba":
        return MambaDecoder
    # elif decoder_name == "transformer_linear":
    #     return LinearAttentionDecoder
    elif decoder_name == "cp_decoder":
        return MistralCompoundDecoder
    elif decoder_name == "patch_mistral":
        return MistralPatchDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
