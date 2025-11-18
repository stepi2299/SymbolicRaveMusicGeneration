import torch
import logging as lg
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from src.constants import ExportFormat

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
        export_format: str = "wav",
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
        self.export_format = ExportFormat(export_format)

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
                logger.info(
                    f"=== [SamplingCallback] Samples at epoch {current_epoch} ==="
                )
                try:
                    paths_to_gen = pl_module.generate(
                        z,
                        folder_path=self.output_dir,
                        file_name=f"sample_{current_epoch}",
                        export_format=self.export_format,
                    )
                    logger.info(
                        f"Generated samples from epoch: {current_epoch} in format: {self.export_format} "
                        f"in path: {paths_to_gen}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed in songs generation on epoch: {current_epoch}, error: {e}"
                    )

            pl_module.train()
