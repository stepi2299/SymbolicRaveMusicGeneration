import hydra
from omegaconf import DictConfig
from pathlib import Path

import torch
from frechet_music_distance import FrechetMusicDistance


@hydra.main(config_path="../../configs", config_name="evaluate_fmd", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(cfg.processing_precision)

    reference_dataset = cfg.reference_dataset_dir
    dataset_path = Path.cwd() / "data" / reference_dataset
    reference_dataset_path = dataset_path.resolve(True)

    testing_dataset_path = Path(cfg.testing_dataset_dir).resolve(True)

    metric = FrechetMusicDistance(verbose=True)
    fdm_score = metric.score(
        reference_dataset=reference_dataset_path.as_posix(),
        test_dataset=testing_dataset_path.as_posix(),
    )
    # fdm_inf_score = metric.score_inf(
    #     reference_dataset=reference_dataset_path.as_posix(),
    #     test_dataset=testing_dataset_path.as_posix(),
    #     min_n=50,
    # )
    print(f"FMD Score: {fdm_score}")


if __name__ == "__main__":
    main()
