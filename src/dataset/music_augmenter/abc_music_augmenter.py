from src.dataset.music_augmenter.base_music_augmenter import BaseMusicAugmenter


class AbcAugmentor(BaseMusicAugmenter):
    def __init__(self, **augmentation_params):
        pass

    def __call__(self, music_data: list[str]) -> list[str]:
        return music_data
