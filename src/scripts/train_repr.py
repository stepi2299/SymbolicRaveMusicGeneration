import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.dataset import get_dataset
from src.models import get_vae_model
from src.competitors.first_stage import FirstStage


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    symbolic_music_dataset = get_dataset(cfg.dataset)
    vae_model = get_vae_model(cfg.model)
    competitor = FirstStage(
        vae=vae_model,
        symbolic_music_dataset=symbolic_music_dataset,
        **cfg.competitor.params,
    )
    trainer = Trainer(callbacks=[], max_epochs=cfg.competitor.epochs)
    trainer.fit(competitor)


if __name__ == "__main__":
    main()
