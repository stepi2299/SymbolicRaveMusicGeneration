from torch.utils.data import DataLoader
from src.dataset.utils import symbolic_collate_fn

from src.dataset.symbolic_music_dataset import SymbolicMusicDataset


def get_dataloader(
    symbolic_music_dataset: SymbolicMusicDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        symbolic_music_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=symbolic_collate_fn,
        num_workers=num_workers,
    )
