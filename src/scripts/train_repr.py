import hydra
from omegaconf import DictConfig

from src.dataset import get_dataset

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    get_dataset(cfg.dataset)

if __name__ == "__main__":
    main()