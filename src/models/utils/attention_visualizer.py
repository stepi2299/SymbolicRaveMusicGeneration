"""
Attention Map Visualizer for symbolic music generation decoders.

Supports:
- GptDecoder: nn.TransformerDecoder with hook-based attention extraction
- TransformerRope: GPTNeoX / CustomGPTNeoXModel with output_attentions=True
- TransformerXL: TransfoXLModel with output_attentions=True
"""

import logging as lg
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.models.sequence_decoder.base_decoder import BaseDecoder
from src.models.sequence_decoder.gpt_decoder import GptDecoder
from src.models.sequence_decoder.transformer_rope import TransformerRope
from src.models.sequence_decoder.transformer_xl import TransformerXL

logger = lg.getLogger(__name__)

# Minimum sequence length to produce a meaningful attention map
_MIN_SEQ_LEN = 4
# Maximum sequence length used during the visualisation forward pass
_VIS_SEQ_LEN = 16


def _set_eager_attn(model: torch.nn.Module) -> None:
    """Set attn_implementation to 'eager' on a HuggingFace model if supported.

    This is required for GPTNeoXModel to return attention weights when
    ``output_attentions=True`` is passed.
    """
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not set eager attn implementation: %s", exc)
    elif hasattr(model, "config"):
        try:
            model.config._attn_implementation = "eager"
        except Exception:  # pragma: no cover
            pass


class AttentionVisualizer:
    """Extracts attention maps from different decoder architectures,
    and saves a grid of heatmaps to disk."""

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_attentions(
        self,
        decoder: BaseDecoder,
        wrapper: "object",  # SimpleWrapper – typed loosely to avoid circular import
        device: torch.device,
    ) -> np.ndarray:
        """Run a forward pass with a minimal BOS-only input and extract
        attention weights from all layers and all heads.

        Dispatches to the correct extraction strategy based on decoder type.

        Args:
            decoder: one of GptDecoder, TransformerRope, or TransformerXL.
            wrapper: a SimpleWrapper instance that owns token_embeddings.
            device: torch device for tensor operations.

        Returns:
            np.ndarray of shape [num_layers, num_heads, seq_len, kv_len].
            For GptDecoder and TransformerRope: kv_len == seq_len.
            For TransformerXL: kv_len == mem_len + seq_len.
        """
        if isinstance(decoder, GptDecoder):
            return self._extract_gpt_attentions(decoder, wrapper, device)
        elif isinstance(decoder, TransformerRope):
            return self._extract_rope_attentions(decoder, wrapper, device)
        elif isinstance(decoder, TransformerXL):
            return self._extract_xl_attentions(decoder, wrapper, device)
        else:
            raise TypeError(f"Unsupported decoder type for attention visualisation: " f"{type(decoder).__name__}")

    def save_plot(
        self,
        attentions: np.ndarray,
        title: str,
        save_path: Path,
        max_layers: int = 4,
        max_heads: int = 4,
        is_xl: bool = False,
    ) -> None:
        """Save a grid of attention heatmaps (rows = layers, cols = heads).

        Args:
            attentions: [num_layers, num_heads, seq_len, kv_len].
            title: suptitle for the figure.
            save_path: output .png file path.
            max_layers: maximum number of layers to display.
            max_heads: maximum number of heads to display.
            is_xl: when True, x-axis is labelled "key/mem position".
        """
        import matplotlib.pyplot as plt  # lazy import

        n_layers = min(attentions.shape[0], max_layers)
        n_heads = min(attentions.shape[1], max_heads)

        fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers), squeeze=False)

        xlabel = "key/mem position" if is_xl else "key position"
        ylabel = "query position"

        for li in range(n_layers):
            for hi in range(n_heads):
                ax = axes[li][hi]
                data = attentions[li, hi]  # [seq_len, kv_len]
                ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
                ax.set_title(f"L{li} H{hi}", fontsize=9)
                if li == n_layers - 1:
                    ax.set_xlabel(xlabel, fontsize=7)
                if hi == 0:
                    ax.set_ylabel(ylabel, fontsize=7)
                ax.tick_params(labelsize=6)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.93, 1.0])

        # Shared colourbar on the right
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved attention map plot to %s", save_path)

    def visualize_and_save(
        self,
        decoder: BaseDecoder,
        wrapper: "object",
        device: torch.device,
        save_path: Path,
    ) -> None:
        """Full pipeline: extract attention maps → plot → save.

        Args:
            decoder: one of GptDecoder, TransformerRope, or TransformerXL.
            wrapper: a SimpleWrapper instance.
            device: torch device for tensor operations.
            save_path: output .png file path.
        """
        logger.info(
            "Starting attention map visualisation for decoder %s",
            type(decoder).__name__,
        )

        attentions = self.extract_attentions(decoder, wrapper, device)

        n_layers = attentions.shape[0]
        n_heads = attentions.shape[1]
        shown_layers = min(n_layers, 4)
        shown_heads = min(n_heads, 4)

        if isinstance(decoder, GptDecoder):
            title = f"GPT Self-Attention Maps " f"(first {shown_layers} layers × {shown_heads} heads)"
            is_xl = False
        elif isinstance(decoder, TransformerRope):
            title = f"RoPE Transformer Attention Maps " f"(first {shown_layers} layers × {shown_heads} heads)"
            is_xl = False
        elif isinstance(decoder, TransformerXL):
            title = f"Transformer-XL Attention Maps " f"(first {shown_layers} layers × {shown_heads} heads)"
            is_xl = True
        else:
            title = f"Attention Maps — {type(decoder).__name__}"
            is_xl = False

        self.save_plot(attentions, title, save_path, is_xl=is_xl)

    # ─────────────────────────────────────────────────────────────��────────────
    # Private extraction helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_gpt_attentions(
        self,
        decoder: GptDecoder,
        wrapper: "object",
        device: torch.device,
    ) -> np.ndarray:
        """Register forward hooks on each TransformerDecoderLayer.self_attn to
        capture attention weights with need_weights=True, average_attn_weights=False.

        Runs one forward pass and returns [num_layers, num_heads, seq_len, seq_len].
        Cleans up patches regardless of exceptions.
        """
        seq_len = max(_MIN_SEQ_LEN, min(decoder.sequence_length, _VIS_SEQ_LEN))
        emb = self._make_input_embedding(wrapper, decoder, seq_len, device)

        captured: dict[int, torch.Tensor] = {}
        patchers = []

        def _make_patched_forward(original_forward, layer_idx: int):
            def patched_forward(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                result = original_forward(*args, **kwargs)
                # result is (attn_output, attn_weights)
                if isinstance(result, tuple | list) and len(result) >= 2:
                    captured[layer_idx] = result[1]
                return result

            return patched_forward

        for i, layer in enumerate(decoder.model.layers):
            patcher = patch.object(
                layer.self_attn,
                "forward",
                new=_make_patched_forward(layer.self_attn.forward, i),
            )
            patcher.start()
            patchers.append(patcher)

        try:
            # Determine conditioning args
            class_idx_tensor, latent_vector = self._conditioning_args(decoder, device)
            decoder.eval()
            decoder(
                emb,
                latent_vector=latent_vector,
                class_idx=class_idx_tensor,
                use_cache=False,
            )
        finally:
            for p in patchers:
                p.stop()

        if not captured:
            raise RuntimeError(
                "No attention weights were captured for GptDecoder. " "Check that hooks are correctly registered."
            )

        # Stack layers in order
        attn_list = [captured[i].cpu().float() for i in range(len(captured))]
        # Each tensor: [batch, num_heads, tgt, src] → squeeze batch
        stacked = torch.stack(attn_list, dim=0)  # [n_layers, batch, heads, tgt, src]
        stacked = stacked.squeeze(1)  # [n_layers, heads, tgt, src]
        return stacked.numpy()

    @torch.no_grad()
    def _extract_rope_attentions(
        self,
        decoder: TransformerRope,
        wrapper: "object",
        device: torch.device,
    ) -> np.ndarray:
        """Call decoder.model with output_attentions=True and stack the results.

        Returns [num_layers, num_heads, seq_len, seq_len].
        """
        seq_len = max(_MIN_SEQ_LEN, min(decoder.sequence_length, _VIS_SEQ_LEN))
        emb = self._make_input_embedding(wrapper, decoder, seq_len, device)

        class_idx_tensor, latent_vector = self._conditioning_args(decoder, device)

        decoder.eval()

        # GPTNeoX requires 'eager' attention implementation to output attention weights
        _set_eager_attn(decoder.model)

        # Build the conditioning embedding if needed for custom model
        if decoder.use_cross_attention or decoder.adaln_norm:
            batch_size = emb.shape[0]
            if decoder._class_conditioning:
                cond_emb = decoder._prepare_class_embedding(class_idx_tensor, batch_size, seq_len)
            else:
                cond_emb = decoder._prepare_latent_embedding(latent_vector, batch_size, seq_len)
            outputs = decoder.model(
                inputs_embeds=emb,
                encoder_hidden_states=cond_emb,
                output_attentions=True,
                return_dict=True,
            )
        else:
            prepared_emb = decoder._prepare_input(emb, latent_vector, class_idx_tensor)
            outputs = decoder.model(
                inputs_embeds=prepared_emb,
                output_attentions=True,
                return_dict=True,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "TransformerRope model returned no attention weights. "
                "Ensure output_attentions=True is supported by this model variant."
            )

        # attentions: tuple of [batch, num_heads, seq_len, seq_len] per layer
        attn_list = [a.cpu().float() for a in outputs.attentions]
        stacked = torch.stack(attn_list, dim=0)  # [n_layers, batch, heads, S, S]
        stacked = stacked.squeeze(1)  # [n_layers, heads, S, S]
        return stacked.numpy()

    @torch.no_grad()
    def _extract_xl_attentions(
        self,
        decoder: TransformerXL,
        wrapper: "object",
        device: torch.device,
    ) -> np.ndarray:
        """Call TransfoXLModel with output_attentions=True and stack the results.

        Note: XL attention dim is [batch, num_heads, seq_len, mem_len + seq_len].

        Returns [num_layers, num_heads, seq_len, mem_len + seq_len].
        """
        seq_len = max(_MIN_SEQ_LEN, min(decoder.sequence_length, _VIS_SEQ_LEN))
        emb = self._make_input_embedding(wrapper, decoder, seq_len, device)

        class_idx_tensor, latent_vector = self._conditioning_args(decoder, device)
        prepared_emb = decoder._prepare_input(emb, latent_vector, class_idx_tensor)

        decoder.eval()
        outputs = decoder.model(
            inputs_embeds=prepared_emb,
            output_attentions=True,
        )

        if outputs.attentions is None:
            raise RuntimeError(
                "TransformerXL model returned no attention weights. " "Ensure output_attentions=True is supported."
            )

        # attentions: tuple of [batch, num_heads, seq_len, mem_len+seq_len] per layer
        attn_list = [a.cpu().float() for a in outputs.attentions]
        stacked = torch.stack(attn_list, dim=0)  # [n_layers, batch, heads, S, S+M]
        stacked = stacked.squeeze(1)  # [n_layers, heads, S, S+M]
        return stacked.numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _make_input_embedding(
        self,
        wrapper: "object",
        decoder: BaseDecoder,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a minimal input embedding of shape [1, seq_len, hidden_size].

        Handles all three wrapper types via duck typing:
        - SimpleWrapper: uses wrapper.token_embeddings (nn.Embedding).
        - CompoundWordWrapper: uses wrapper.cp_embedding with a multi-attribute
          BOS tensor of shape [1, seq_len, num_attrs].
        - PatchWrapper: uses wrapper._patch_embedding with a patch tensor of
          shape [1, seq_len, patch_size].

        Args:
            wrapper: any BaseWrapper instance.
            decoder: the target decoder.
            seq_len: number of BOS positions to embed.
            device: torch device.

        Returns:
            torch.Tensor of shape [1, seq_len, hidden_size].
        """
        # SimpleWrapper
        if hasattr(wrapper, "token_embeddings"):
            bos_seq = torch.full((1, seq_len), decoder.bos_token, dtype=torch.long, device=device)
            return wrapper.token_embeddings(bos_seq)  # [1, seq_len, hidden]

        # CompoundWordWrapper — input is [B, L, num_attrs]; fill all attrs with bos
        if hasattr(wrapper, "cp_embedding"):
            from src.constants import CP_WORD_INDEXES

            num_attrs = len(CP_WORD_INDEXES)
            bos_seq = torch.full(
                (1, seq_len, num_attrs),
                decoder.bos_token,
                dtype=torch.long,
                device=device,
            )
            return wrapper.cp_embedding(bos_seq)  # [1, seq_len, hidden]

        # PatchWrapper — input is [B, num_patches, patch_size]; fill with bos
        if hasattr(wrapper, "_patch_embedding"):
            patch_size = decoder.patch_size
            bos_seq = torch.full(
                (1, seq_len, patch_size),
                decoder.bos_token,
                dtype=torch.long,
                device=device,
            )
            return wrapper._patch_embedding(bos_seq)  # [1, seq_len, hidden]

        raise TypeError(
            f"Unsupported wrapper type for embedding extraction: "
            f"{type(wrapper).__name__}. Expected SimpleWrapper, "
            "CompoundWordWrapper, or PatchWrapper."
        )

    def _conditioning_args(
        self,
        decoder: BaseDecoder,
        device: torch.device,
    ) -> tuple:
        """Return (class_idx_tensor, latent_vector) appropriate for the decoder.

        Returns zero-valued tensors when conditioning is required.
        """
        if decoder._class_conditioning:
            class_idx_tensor = torch.zeros(1, dtype=torch.long, device=device)
            latent_vector = None
        elif decoder.latent_dim is not None:
            class_idx_tensor = None
            latent_vector = torch.zeros(1, decoder.latent_dim, dtype=torch.float, device=device)
        else:
            class_idx_tensor = None
            latent_vector = None
        return class_idx_tensor, latent_vector
