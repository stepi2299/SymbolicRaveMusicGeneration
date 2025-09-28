import torch

from src.loss import ElboLoss


class TestElboLoss:
    @staticmethod
    def test_elbo_loss_inference():
        """Basic sanity check: loss runs and returns expected keys."""
        vocab_size = 5
        batch, length = 2, 3
        loss_fn = ElboLoss(vocab_size)

        org_data = torch.randint(0, vocab_size, (batch, length))
        reconstruction = torch.randn(batch, length, vocab_size, requires_grad=True)
        mu = torch.zeros(batch, 4)       # latent dim = 4
        logvar = torch.zeros(batch, 4)

        loss, metrics = loss_fn(org_data, reconstruction, mu, logvar)

        assert isinstance(loss, torch.Tensor)
        assert set(metrics.keys()) == {"recon_loss", "kl_div"}
        # By default with mu=0 and logvar=0, KL should be ~0
        assert torch.allclose(metrics["kl_div"], torch.tensor(0.0), atol=1e-6)

    @staticmethod
    def test_elbo_perfect_reconstruction_zero_kl():
        """If reconstruction is perfect and KL=0, total loss ~0."""
        vocab_size = 3
        batch, length = 2, 2
        loss_fn = ElboLoss(vocab_size)

        # org_data are indices
        org_data = torch.tensor([[0, 1], [2, 1]])
        # reconstruction logits: put huge positive score on the correct class
        reconstruction = torch.full((batch, length, vocab_size), -10.0)
        for i in range(batch):
            for j in range(length):
                reconstruction[i, j, org_data[i, j]] = 10.0

        mu = torch.zeros(batch, 2)
        logvar = torch.zeros(batch, 2)

        loss, metrics = loss_fn(org_data, reconstruction, mu, logvar)

        assert metrics["recon_loss"] < 1e-3
        assert metrics["kl_div"].abs() < 1e-6
        assert loss < 1e-3

    @staticmethod
    def test_elbo_perfect_reconstruction_high_kl():
        """Perfect reconstruction but large KL should make loss dominated by KL."""
        vocab_size = 3
        batch, length = 2, 2
        loss_fn = ElboLoss(vocab_size)

        org_data = torch.tensor([[0, 1], [2, 1]])
        reconstruction = torch.full((batch, length, vocab_size), -10.0)
        for i in range(batch):
            for j in range(length):
                reconstruction[i, j, org_data[i, j]] = 10.0

        # Large mu -> large KL divergence
        mu = torch.ones(batch, 2) * 10.0
        logvar = torch.zeros(batch, 2)

        loss, metrics = loss_fn(org_data, reconstruction, mu, logvar)

        assert metrics["recon_loss"] < 1e-3  # still perfect reconstruction
        assert metrics["kl_div"] > 50.0      # KL should be large
        assert torch.isclose(loss, metrics["kl_div"], rtol=1e-3)

    @staticmethod
    def test_elbo_bad_reconstruction():
        """Bad reconstruction should lead to high reconstruction loss."""
        vocab_size = 3
        batch, length = 2, 2
        loss_fn = ElboLoss(vocab_size)

        org_data = torch.tensor([[0, 1], [2, 1]])
        # reconstruction logits: uniform predictions
        reconstruction = torch.zeros(batch, length, vocab_size)

        mu = torch.zeros(batch, 2)
        logvar = torch.zeros(batch, 2)

        loss, metrics = loss_fn(org_data, reconstruction, mu, logvar)

        # For uniform logits, CE loss ~ log(vocab_size)
        expected_recon = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        assert torch.allclose(metrics["recon_loss"], expected_recon, atol=1e-2)
        assert metrics["kl_div"].abs() < 1e-6
        assert torch.allclose(loss, expected_recon, atol=1e-2)

