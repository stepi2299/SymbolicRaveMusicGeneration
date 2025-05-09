import torch
from torch.utils.data import DataLoader
from src.dataset.dataloader import get_dataloader


class TestDataLoader:
    @staticmethod
    def test_get_dataloader_parameters(midi_symbolic_dataset):
        batch_size = 4
        shuffle = True
        num_workers = 2

        dataloader = get_dataloader(
            symbolic_music_dataset=midi_symbolic_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        assert isinstance(
            dataloader, DataLoader
        ), "get_dataloader should return a DataLoader."
        # Check DataLoader parameters
        assert dataloader.batch_size == batch_size, f"Expected batch_size={batch_size}"
        assert (
            dataloader.num_workers == num_workers
        ), f"Expected num_workers={num_workers}"

    # @staticmethod
    # def test_get_dataloader_iteration(midi_symbolic_dataset):
    #     batch_size = 2
    #
    #     dataloader = get_dataloader(
    #         symbolic_music_dataset=midi_symbolic_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,  # Turn off shuffle so we know the order
    #         num_workers=0,
    #     )
    #
    #     all_batches = list(dataloader)
    #     assert len(all_batches) == 2, f"Expected 2 batches, got {len(all_batches)}."
    #
    #     for i, batch in enumerate(all_batches):
    #         assert isinstance(
    #             batch, list
    #         ), f"Expected batch to be a list, got {type(batch)}"
    #         if i == 0:
    #             assert (
    #                 len(batch) == batch_size
    #             ), f"Expected batch_size={batch_size}, got {len(batch)}"
    #         else:
    #             assert (
    #                 len(batch) == 1
    #             ), f"Last batch, expected batch_size=1, got {len(batch)}"
    #         for song in batch:
    #             assert isinstance(
    #                 song, torch.Tensor
    #             ), f"Expected song to be a Tensor, got {type(song)}"
    #             for path in song:
    #                 assert isinstance(
    #                     path, torch.Tensor
    #                 ), f"Expected path to be a Tensor, got {type(path)}"
    #                 for note in path:
    #                     assert isinstance(
    #                         note, torch.Tensor
    #                     ), f"Expected note to be an Tensor, got {type(note)}"
