"""Unit tests for PositionalEncodingVisualizer."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.utils.pe_visualizer import PositionalEncodingVisualizer


# ── Shared tiny-config constants ─────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEQ_LEN = 16
VOCAB_SIZE = 256
NUM_CLASSES = 5
DEVICE = torch.device("cpu")

# Common kwargs shared by all decoder constructors
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
def xl_decoder():
    from src.models.sequence_decoder.transformer_xl import TransformerXL

    return TransformerXL(
        **_COMMON,
        mem_len=SEQ_LEN,
        attention_dropout=0.0,
    ).to(DEVICE).eval()


@pytest.fixture
def visualizer():
    return PositionalEncodingVisualizer()


# ── GptDecoder tests ──────────────────────────────────────────────────────────

class TestGptDecoderPE:
    def test_extract_pe_shape(self, visualizer, gpt_decoder):
        pe = visualizer.extract_pe(gpt_decoder, SEQ_LEN, DEVICE)
        assert isinstance(pe, np.ndarray)
        assert pe.shape == (SEQ_LEN, HIDDEN_SIZE)

    def test_tsne_shape(self, visualizer, gpt_decoder):
        pe = visualizer.extract_pe(gpt_decoder, SEQ_LEN, DEVICE)
        tsne = visualizer.run_tsne(pe)
        assert tsne.shape == (SEQ_LEN, 2)

    def test_visualize_and_save(self, visualizer, gpt_decoder, tmp_path):
        save_path = tmp_path / "pe_gpt.png"
        visualizer.visualize_and_save(
            decoder=gpt_decoder,
            sequence_length=SEQ_LEN,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for GptDecoder"


# ── TransformerRope tests ─────────────────────────────────────────────────────

class TestTransformerRopePE:
    def test_extract_pe_shape(self, visualizer, rope_decoder):
        pe = visualizer.extract_pe(rope_decoder, SEQ_LEN, DEVICE)
        assert isinstance(pe, np.ndarray)
        assert pe.shape[0] == SEQ_LEN
        # second dim = 2 * rotary_dim; just check it's > 0
        assert pe.shape[1] > 0

    def test_tsne_shape(self, visualizer, rope_decoder):
        pe = visualizer.extract_pe(rope_decoder, SEQ_LEN, DEVICE)
        tsne = visualizer.run_tsne(pe)
        assert tsne.shape == (SEQ_LEN, 2)

    def test_visualize_and_save(self, visualizer, rope_decoder, tmp_path):
        save_path = tmp_path / "pe_rope.png"
        visualizer.visualize_and_save(
            decoder=rope_decoder,
            sequence_length=SEQ_LEN,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for TransformerRope"


# ── TransformerXL tests ───────────────────────────────────────────────────────

class TestTransformerXLPE:
    def test_extract_pe_shape(self, visualizer, xl_decoder):
        pe = visualizer.extract_pe(xl_decoder, SEQ_LEN, DEVICE)
        assert isinstance(pe, np.ndarray)
        # n_layers * 2 rows
        assert pe.shape[0] == NUM_LAYERS * 2
        assert pe.shape[1] > 0

    def test_tsne_shape(self, visualizer, xl_decoder):
        pe = visualizer.extract_pe(xl_decoder, SEQ_LEN, DEVICE)
        tsne = visualizer.run_tsne(pe)
        assert tsne.shape == (NUM_LAYERS * 2, 2)

    def test_visualize_and_save(self, visualizer, xl_decoder, tmp_path):
        save_path = tmp_path / "pe_xl.png"
        visualizer.visualize_and_save(
            decoder=xl_decoder,
            sequence_length=SEQ_LEN,
            device=DEVICE,
            save_path=save_path,
        )
        assert save_path.exists(), "PNG file was not created for TransformerXL"


# ── Edge-case: perplexity auto-reduction ─────────────────────────────────────

class TestTsnePerplexityEdgeCase:
    def test_perplexity_reduced_when_n_small(self, visualizer):
        """When N < default perplexity, run_tsne should not crash."""
        small_embeddings = np.random.randn(5, 64).astype(np.float32)
        result = visualizer.run_tsne(small_embeddings, perplexity=30)
        assert result.shape == (5, 2)

    def test_very_small_n(self, visualizer):
        """When N < 4, fall back to PCA-based 2D projection."""
        tiny_embeddings = np.random.randn(3, 64).astype(np.float32)
        result = visualizer.run_tsne(tiny_embeddings, perplexity=30)
        assert result.shape == (3, 2)

