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
        output_dir: str,
        sample_interval: int = 1,
        num_samples: int = 2,
        export_format: str = "wav",
    ):
        """
        sample_interval: epochs between sample generations
        num_samples: number of samples to generate each time
        """
        super().__init__()
        self.sample_interval = sample_interval
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
                logger.info(
                    f"=== [SamplingCallback] Samples at epoch {current_epoch} ==="
                )
                try:
                    pl_module.generate(
                        num_samples = self.num_samples,
                        folder_path=self.output_dir,
                        file_name=f"sample_{current_epoch}",
                        export_format=self.export_format,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed in songs generation on epoch: {current_epoch}, error: {e}"
                    )

            pl_module.train()
