import torch

from pytorch_lightning.callbacks import Callback


class SamplingCallback(Callback):
    """
    Generates samples from the VAE's decoder at the end of each epoch.
    """

    def __init__(self,
                 latent_dim: int,
                 sample_interval: int = 1,
                 num_samples: int = 2):
        """
        sample_interval: epochs between sample generations
        latent_dim: dimension of the VAE latent space
        num_samples: number of samples to generate each time
        """
        super().__init__()
        self.sample_interval = sample_interval
        self.latent_dim = latent_dim
        self.num_samples = num_samples

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        current_epoch = trainer.current_epoch
        if current_epoch % self.sample_interval == 0:
            # Switch to eval mode
            pl_module.eval()
            with torch.no_grad():
                # Sample random latent vectors
                z = torch.randn(self.num_samples, self.latent_dim, device=pl_module.device)
                generated = pl_module.decode(z)

            print(f"=== [SamplingCallback] Samples at epoch {current_epoch} ===")
            print("Generated shape:", generated.shape)
            # TODO do sth with this generation
            # Switch back to train mode if needed
            pl_module.train()
