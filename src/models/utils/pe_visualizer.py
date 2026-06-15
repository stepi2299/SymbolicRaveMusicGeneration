"""
Positional Encoding Visualizer for symbolic music generation decoders.

Supports:
- GptDecoder: learned absolute positional embeddings
- TransformerRope: Rotary Position Embeddings (RoPE)
- TransformerXL: relative positional bias tensors
"""

import logging as lg
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.sequence_decoder.base_decoder import BaseDecoder
from src.models.sequence_decoder.gpt_decoder import GptDecoder
from src.models.sequence_decoder.transformer_rope import TransformerRope
from src.models.sequence_decoder.transformer_xl import TransformerXL

logger = lg.getLogger(__name__)


class PositionalEncodingVisualizer:
    """Extracts positional encodings from different decoder architectures,
    reduces them with t-SNE, and saves scatter plots to disk."""

    def extract_pe(
        self,
        decoder: "BaseDecoder",
        sequence_length: int,
        device: torch.device,
    ) -> np.ndarray:
        """Dispatch to the correct extraction strategy based on decoder type.

        Returns:
            np.ndarray of shape [N, D] where N is the number of positions/heads.
        """

        if isinstance(decoder, GptDecoder):
            return self._extract_learned_pe(decoder, sequence_length, device)
        elif isinstance(decoder, TransformerRope):
            return self._extract_rope_pe(decoder, sequence_length, device)
        elif isinstance(decoder, TransformerXL):
            return self._extract_xl_pe(decoder)
        else:
            raise TypeError(f"Unsupported decoder type for PE visualization: {type(decoder).__name__}")

    @torch.no_grad()
    def _extract_learned_pe(
        self,
        decoder: "GptDecoder",
        sequence_length: int,
        device: torch.device,
    ) -> np.ndarray:
        """Extract nn.Embedding weights for all positions.

        Returns:
            np.ndarray of shape [seq_len, hidden_size].
        """
        position_idx = torch.arange(sequence_length, dtype=torch.long, device=device)
        pe = decoder.positional_embeddings(position_idx)  # [seq_len, hidden_size]
        return pe.cpu().float().numpy()

    @torch.no_grad()
    def _extract_rope_pe(
        self,
        decoder: "TransformerRope",
        sequence_length: int,
        device: torch.device,
    ) -> np.ndarray:
        """Compute RoPE cos/sin and concatenate them as the position representation.

        Returns:
            np.ndarray of shape [seq_len, 2 * rotary_dim].
        """
        # Create a dummy hidden_states tensor with the correct shape
        # rotary_emb lives on the model (GPTNeoXModel or CustomGPTNeoXModel)
        hidden_size = decoder.hidden_size
        dummy = torch.zeros(1, sequence_length, hidden_size, device=device)
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

        cos, sin = decoder.model.rotary_emb(dummy, position_ids)
        # cos/sin: [1, seq_len, rotary_dim]
        cos = cos.squeeze(0)  # [seq_len, rotary_dim]
        sin = sin.squeeze(0)  # [seq_len, rotary_dim]
        pe = torch.cat([cos, sin], dim=-1)  # [seq_len, 2*rotary_dim]
        return pe.cpu().float().numpy()

    @torch.no_grad()
    def _extract_xl_pe(self, decoder: "TransformerXL") -> np.ndarray:
        """Extract and average r_w_bias and r_r_bias across all layers.

        Returns:
            np.ndarray of shape [n_layers * 2, head_dim * n_heads] — one row per
            (layer, bias_type) combination, useful for t-SNE over layers.
        """
        rows = []
        for layer in decoder.model.layers:
            attn = layer.dec_attn
            # r_w_bias: [n_heads, head_dim]
            r_w = attn.r_w_bias.detach().cpu().float()
            r_r = attn.r_r_bias.detach().cpu().float()
            # Flatten to 1-D vector per bias type
            rows.append(r_w.reshape(-1).numpy())
            rows.append(r_r.reshape(-1).numpy())
        return np.stack(rows, axis=0)  # [n_layers * 2, n_heads * head_dim]

    def run_tsne(
        self,
        embeddings: np.ndarray,
        perplexity: int = 30,
        n_iter: int = 1000,
    ) -> np.ndarray:
        """Run sklearn TSNE on embeddings.

        Args:
            embeddings: [N, D] array.
            perplexity: t-SNE perplexity. Automatically reduced if N is too small.
            n_iter: number of optimisation iterations.

        Returns:
            np.ndarray of shape [N, 2].
        """

        n_samples = embeddings.shape[0]
        effective_perplexity = min(perplexity, n_samples - 1)
        if effective_perplexity != perplexity:
            logger.warning(
                "Reducing t-SNE perplexity from %d to %d because N=%d",
                perplexity,
                effective_perplexity,
                n_samples,
            )

        # When N is very small (< 4), t-SNE is degenerate; fall back to PCA-like 2D slice
        if n_samples < 4:
            logger.warning(
                "Too few samples (%d) for t-SNE; returning first two PCA components.",
                n_samples,
            )

            n_components = min(2, n_samples)
            pca = PCA(n_components=n_components)
            result = pca.fit_transform(embeddings)
            if result.shape[1] < 2:
                result = np.pad(result, ((0, 0), (0, 2 - result.shape[1])))
            return result

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            max_iter=n_iter,
            random_state=42,
        )
        return tsne.fit_transform(embeddings)

    def save_plot(
        self,
        tsne_result: np.ndarray,
        labels: list[str],
        title: str,
        save_path: Path,
    ) -> None:
        """Save a matplotlib scatter plot of t-SNE results.

        Colors points by their index (position/layer). Uses the 'viridis' colormap.

        Args:
            tsne_result: [N, 2] array from run_tsne.
            labels: list of length N with per-point label strings (used for colorbar ticks).
            title: plot title string.
            save_path: file path to save the .png image.
        """

        n = tsne_result.shape[0]
        color_values = np.arange(n)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=color_values,
            cmap="viridis",
            s=40,
            alpha=0.85,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax)
        if labels:
            # Show a sparse set of tick labels so they don't overlap
            tick_indices = np.linspace(0, n - 1, min(10, n), dtype=int)
            cbar.set_ticks(tick_indices)
            cbar.set_ticklabels([labels[i] for i in tick_indices])

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1", fontsize=11)
        ax.set_ylabel("t-SNE dim 2", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PE t-SNE plot to %s", save_path)

    def visualize_and_save(
        self,
        decoder: "BaseDecoder",
        sequence_length: int,
        device: torch.device,
        save_path: Path,
    ) -> None:
        """Full pipeline: extract PE → run t-SNE → plot → save.

        Args:
            decoder: one of GptDecoder, TransformerRope, or TransformerXL.
            sequence_length: number of positions to extract.
            device: torch device for tensor operations.
            save_path: output .png file path.
        """

        logger.info(
            "Starting PE visualization for decoder %s, sequence_length=%d",
            type(decoder).__name__,
            sequence_length,
        )

        embeddings = self.extract_pe(decoder, sequence_length, device)
        tsne_result = self.run_tsne(embeddings)

        n = embeddings.shape[0]

        if isinstance(decoder, GptDecoder):
            title = "Learned Absolute PE (t-SNE)"
            labels = [str(i) for i in range(n)]
        elif isinstance(decoder, TransformerRope):
            title = "RoPE Embeddings (t-SNE) — cos∥sin"
            labels = [str(i) for i in range(n)]
        elif isinstance(decoder, TransformerXL):
            title = "Transformer-XL Relative PE Biases (t-SNE)"
            n_layers = n // 2
            labels = []
            for layer_idx in range(n_layers):
                labels.append(f"L{layer_idx}_rw")
                labels.append(f"L{layer_idx}_rr")
        else:
            title = f"PE t-SNE — {type(decoder).__name__}"
            labels = [str(i) for i in range(n)]

        self.save_plot(tsne_result, labels, title, save_path)
