import torch
import logging as lg
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from src.utils import save_midi_song_from_string

logger = lg.getLogger(__name__)


class SamplingCallback(Callback):
    """
    Generates samples from the VAE's decoder at the end of each epoch.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dir: str,
        sample_interval: int = 1,
        num_samples: int = 2,
    ):
        """
        sample_interval: epochs between sample generations
        latent_dim: dimension of the VAE latent space
        num_samples: number of samples to generate each time
        """
        super().__init__()
        self.sample_interval = sample_interval
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        output_dir = Path(output_dir)
        self.output_dir = output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        current_epoch = trainer.current_epoch
        if current_epoch % self.sample_interval == 0:
            # Switch to eval mode
            pl_module.eval()
            with torch.no_grad():
                # Sample random latent vectors
                z = torch.randn(
                    self.num_samples, self.latent_dim, device=pl_module.device
                )
                generated = pl_module.generate(z)

            logger.info(f"=== [SamplingCallback] Samples at epoch {current_epoch} ===")
            logger.info(
                f"Generated songs: {len(generated)}",
            )
            for i, song in enumerate(generated):
                try:
                    file_name = self.output_dir / f"sample_{current_epoch}_{i + 1}.mid"
                    save_midi_song_from_string(song, file_name)
                    logger.info(f"Sample {i + 1} successfully saved to {file_name}")
                except Exception as e:
                    logger.error(f"Error saving sample {i + 1}: {e}")
                    continue

            pl_module.train()
