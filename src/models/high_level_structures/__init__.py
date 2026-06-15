import inspect

from src.models.high_level_structures.base_high_level_model import BaseHighLevelModel
from src.models.high_level_structures.sequence_model import SequenceModel
from src.models.high_level_structures.vae import VAE

registered_models = {
    created_class.__name__: created_class
    for created_class in BaseHighLevelModel.__subclasses__()
    if not inspect.isabstract(created_class)
}

__all__ = ["BaseHighLevelModel", "VAE", "SequenceModel", "registered_models"]
