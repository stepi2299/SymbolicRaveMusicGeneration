import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type
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
        self._cache_params = None

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
        intermediate_ff_size: int,
        dropout: float,
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
        )
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_size)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.positional_embeddings = nn.Embedding(
            self.sequence_length, self.hidden_size
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=intermediate_ff_size,
            dropout=dropout,
            norm_first=True,
        )
        self.model = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)

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
        intermediate_ff_size: int,
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
        )

        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        config = TransfoXLConfig(
            d_model=self.hidden_size,
            n_head=self.num_heads,
            n_layer=num_decoder_layers,
            d_inner=intermediate_ff_size,
            vocab_size=vocab_size,
            mem_len=mem_len,  # Length of the memory
            dropout=dropout,
            # Important! Disable adaptive input and output embeddings
            adaptive=False,
            div_val=1,  # No division of embedding size,
            pre_lnorm=True,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
        )

        self.model = TransfoXLModel(config)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self._memory = None

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # input_sequence: [Batch, Seq_Len]
        batch_size, current_sequence_len = input_sequence.shape
        token_emb = self.token_embeddings(input_sequence)

        # [B, Latent] -> [B, Hidden]
        latent_proj = self.latent_proj(latent_vector)

        # [B, 1, Hidden] -> [B, Seq_Len, Hidden]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        decoder_inputs_embeds = token_emb + latent_proj_expanded

        outputs = self.model(inputs_embeds=decoder_inputs_embeds, mems=self._memory)
        last_hidden_state = outputs.last_hidden_state  # [B, Seq_Len, Hidden]
        self._memory = outputs.mems

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
        eos_token: int,
        pad_token: int,
        bos_token: int,
        rotary_pct: float,
        rotary_emb_base: int,
        intermediate_ff_size: int,
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
        )

        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        config = GPTNeoXConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_decoder_layers,
            intermediate_size=intermediate_ff_size,
            vocab_size=vocab_size,
            dropout_rate=dropout,
            max_position_embeddings=sequence_length,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )

        self.model = GPTNeoXModel(config)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,  # set True during inference
    ) -> torch.Tensor:
        # input_sequence: [Batch, Seq_Len]
        batch_size, current_sequence_len = input_sequence.shape
        token_emb = self.token_embeddings(input_sequence)

        # [B, Latent] -> [B, Hidden]
        latent_proj = self.latent_proj(latent_vector)

        # [B, 1, Hidden] -> [B, Seq_Len, Hidden]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        decoder_inputs_embeds = token_emb + latent_proj_expanded

        outputs = self.model(
            inputs_embeds=decoder_inputs_embeds,
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
        num_hidden_layers: int,
        attention_dropout: float,
        rope_theta: float,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        intermediate_ff_size: int,
        sliding_window: int,
        num_key_value_heads: int,
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
        )
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // 64
        config = MistralConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_ff_size,
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

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,  # set True during inference
    ) -> torch.Tensor:
        batch_size, current_sequence_len = input_sequence.shape

        token_emb = self.token_embeddings(input_sequence)  # [B, L, D]

        latent_proj = self.latent_proj(latent_vector)  # [B, D]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        decoder_inputs_embeds = token_emb + latent_proj_expanded

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
        num_hidden_layers: int,
        eos_token: int,
        pad_token: int,
        bos_token: int,
        expand: int,
        conv_kernel: int,
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
        )
        self.hidden_size = hidden_size
        config = MambaConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            expand=expand,
            conv_kernel=conv_kernel,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.pad_token,
            use_cache=False,
        )
        self.model = MambaModel(config)

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self._cache_params = None  # Stan ukryty SSM

    def forward(
        self,
        latent_vector: torch.Tensor,
        input_sequence: torch.Tensor,
        *,
        recv_in_music_form: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, current_sequence_len = input_sequence.shape

        token_emb = self.token_embeddings(input_sequence)  # [B, L, D]

        latent_proj = self.latent_proj(latent_vector)  # [B, D]
        latent_proj_expanded = latent_proj.unsqueeze(1).expand(
            batch_size, current_sequence_len, self.hidden_size
        )
        decoder_inputs_embeds = token_emb + latent_proj_expanded

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
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}, possible: ['simple_conv']")
