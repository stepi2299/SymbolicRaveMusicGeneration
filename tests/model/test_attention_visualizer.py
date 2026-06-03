"""Unit tests for AttentionVisualizer."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.utils.attention_visualizer import AttentionVisualizer
from src.models.decoder_wrappers.simple_wrapper import SimpleWrapper

# ── Shared tiny-config constants ─────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEQ_LEN = 16
VOCAB_SIZE = 256
NUM_CLASSES = 5
DEVICE = torch.device("cpu")

_COMMON = dict(
    latent_dim=None,
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    sequence_length=SEQ_LEN,
    patch_size=4,
    num_decoder_layers=NUM_LAYERS,
    pad_token=0,
    bos_token=1,
    eos_token=2,
    dropout=0.0,
    conditioning="add",
    class_conditioning=True,
    num_classes=NUM_CLASSES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gpt_decoder():
    from src.models.sequence_decoder.gpt_decoder import GptDecoder
    return GptDecoder(**_COMMON).to(DEVICE).eval()


@pytest.fixture(scope="module")
def gpt_wrapper(gpt_decoder):
    return SimpleWrapper(gpt_decoder, input_dropout_rate=0.0).to(DEVICE).eval()


@pytest.fixture(scope="module")
def rope_decoder():
    from src.models.sequence_decoder.transformer_rope import TransformerRope
    return TransformerRope(
        **_COMMON,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        use_cross_attention=False,
        adaln_norm=False,
        condition_layer_size=HIDDEN_SIZE,
    ).to(DEVICE).eval()


@pytest.fixture(scope="module")
def rope_wrapper(rope_decoder):
    return SimpleWrapper(rope_decoder, input_dropout_rate=0.0).to(DEVICE).eval()


@pytest.fixture(scope="module")
def xl_decoder():
    from src.models.sequence_decoder.transformer_xl import TransformerXL
    return TransformerXL(
        **_COMMON,
        mem_len=SEQ_LEN,
        attention_dropout=0.0,
    ).to(DEVICE).eval()


@pytest.fixture(scope="module")
def xl_wrapper(xl_decoder):
    return SimpleWrapper(xl_decoder, input_dropout_rate=0.0).to(DEVICE).eval()


@pytest.fixture
def visualizer():
    return AttentionVisualizer()


# ── GptDecoder tests ──────────────────────────────────────────────────────────

class TestGptDecoderAttention:
    def test_extract_attentions_shape(self, visualizer, gpt_decoder, gpt_wrapper):
        attn = visualizer.extract_attentions(gpt_decoder, gpt_wrapper, DEVICE)
        assert isinstance(attn, np.ndarray)
        num_layers, num_heads, seq_len, kv_len = attn.shape
        assert num_layers == NUM_LAYERS
        assert num_heads == HIDDEN_SIZE // 64
        assert seq_len == kv_len  # square attention for GPT

    def test_values_in_range(self, visualizer, gpt_decoder, gpt_wrapper):
        attn = visualizer.extract_attentions(gpt_decoder, gpt_wrapper, DEVICE)
        assert attn.min() >= -1e-5
        assert attn.max() <= 1 + 1e-5

    def test_visualize_and_save(self, visualizer, gpt_decoder, gpt_wrapper, tmp_path):
        save_path = tmp_path / "attn_gpt.png"
        visualizer.visualize_and_save(
            decoder=gpt_decoder,
            wrapper=gpt_wrapper,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for GptDecoder"


# ── TransformerRope tests ─────────────────────────────────────────────────────

class TestTransformerRopeAttention:
    def test_extract_attentions_shape(self, visualizer, rope_decoder, rope_wrapper):
        attn = visualizer.extract_attentions(rope_decoder, rope_wrapper, DEVICE)
        assert isinstance(attn, np.ndarray)
        num_layers, num_heads, seq_len, kv_len = attn.shape
        assert num_layers == NUM_LAYERS
        assert seq_len == kv_len  # square attention for RoPE

    def test_visualize_and_save(self, visualizer, rope_decoder, rope_wrapper, tmp_path):
        save_path = tmp_path / "attn_rope.png"
        visualizer.visualize_and_save(
            decoder=rope_decoder,
            wrapper=rope_wrapper,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for TransformerRope"


# ── TransformerXL tests ───────────────────────────────────────────────────────

class TestTransformerXLAttention:
    def test_extract_attentions_shape(self, visualizer, xl_decoder, xl_wrapper):
        attn = visualizer.extract_attentions(xl_decoder, xl_wrapper, DEVICE)
        assert isinstance(attn, np.ndarray)
        assert attn.ndim == 4
        num_layers, num_heads, seq_len, kv_len = attn.shape
        assert num_layers == NUM_LAYERS
        # kv_len >= seq_len due to mem_len
        assert kv_len >= seq_len

    def test_visualize_and_save(self, visualizer, xl_decoder, xl_wrapper, tmp_path):
        save_path = tmp_path / "attn_xl.png"
        visualizer.visualize_and_save(
            decoder=xl_decoder,
            wrapper=xl_wrapper,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for TransformerXL"

