from pathlib import Path

import torch
from miditok import REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
from symusic import Score


class Tokenizer:
    TOKENIZER_MAPPER = {
        "remi": REMI,
        "midilike": MIDILike,
        "tsd": TSD,
        "structured": Structured,
        "cpword": CPWord,
        "octuple": Octuple,
        "mumidi": MuMIDI,
        "mmm": MMM,
    }

    def __init__(self, tokenizer_name: str, training_model: str, vocab_size: int):
        self.tokenizer_name = tokenizer_name
        self.training_model = training_model
        self.vocab_size = vocab_size
        if tokenizer_name not in self.TOKENIZER_MAPPER:
            raise ValueError(
                f"Tokenizer {tokenizer_name} not supported, supported tokenizers "
                f"are {self.TOKENIZER_MAPPER.keys()}"
            )
        self.tokenizer = self.TOKENIZER_MAPPER[tokenizer_name]()
        self._is_trained = False

    def tokenize(self, text: Score) -> torch.Tensor:
        if not self._is_trained:
            raise ValueError("Tokenizer is not trained")
        tok_sequence = self.tokenizer.encode(text)
        # TODO How to properly make tensors from tokens
        return torch.tensor(tok_sequence)

    def detokenize(self, tensor_tokens: torch.Tensor) -> Score:
        if not self._is_trained:
            raise ValueError("Tokenizer is not trained")
        # TODO how to make tokens from tensors
        return self.tokenizer.decode(tensor_tokens)

    def train(self, files_paths: list[Path]) -> None:
        self.tokenizer.train(
            vocab_size=self.vocab_size,
            files_paths=files_paths,
            model=self.training_model,
        )
        self._is_trained = True

    def save(self) -> None:
        raise NotImplementedError("Tokenizer saving is not implemented yet")

    def load(self) -> None:
        raise NotImplementedError("Tokenizer loading is not implemented yet")

    @property
    def is_trained(self):
        return self._is_trained
