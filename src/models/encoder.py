import torch
from torch import nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Type
import logging as lg

from transformers import BertConfig
from src.models.clamp2 import PatchEncoder, CLaMP2Model
from src.utils import download_file

logger = lg.getLogger(__name__)


class Encoder(nn.Module, ABC):
    def __init__(self, latent_dim: int, name: str) -> None:
        # TODO think about accelerator
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name

    @abstractmethod
    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO check input data to encoder (symbolic music)
        # TODO Possible handle in some cases from symbolic to tensor (Tokenizer?)
        pass


class DummyEncoder(Encoder):
    def __init__(self, latent_dim: int) -> None:
        super(DummyEncoder, self).__init__(latent_dim=latent_dim)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(data), self.latent_dim), torch.zeros(
            data.shape[0], self.latent_dim
        )


# class SimpleConvEncoder(Encoder):
# def __init__(
#     self, input_shape: Tuple[int, int, int], latent_dim: int, hidden_dim: int = 128
# ) -> None:
#     super(SimpleConvEncoder, self).__init__(latent_dim=latent_dim)
#     self.input_shape = input_shape
#     self.latent_dim = latent_dim
#     self.hidden_dim = hidden_dim
#
#     self.conv1 = nn.Conv2d(
#         in_channels=input_shape[0],
#         out_channels=32,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#     )
#     self.conv2 = nn.Conv2d(
#         in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
#     )
#     self.conv3 = nn.Conv2d(
#         in_channels=64, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1
#     )
#     self.fc_mu = nn.Linear(hidden_dim * input_shape[1] * input_shape[2], latent_dim)
#     self.fc_log_variance = nn.Linear(
#         hidden_dim * input_shape[1] * input_shape[2], latent_dim
#     )
#     self.relu = nn.ReLU()
#
# def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     x = self.conv1(data)
#     x = self.relu(x)
#     x = self.conv2(x)
#     x = self.relu(x)
#     x = self.conv3(x)
#     x = self.relu(x)
#     x = x.view(-1, self.hidden_dim * self.input_shape[1] * self.input_shape[2])
#     mu = self.fc_mu(x)
#     log_variance = self.fc_log_variance(x)
#     return mu, log_variance


class Clamp2Encoder(Encoder):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        patch_length: int,
        patch_size: int,
        latent_dim: int,
        weights_url: str,
    ) -> None:
        super(Clamp2Encoder, self).__init__(latent_dim=latent_dim, name="clamp2")
        self.weights_path = Path("weights/clamp2")
        self.weights_path.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.weights_path / "clamp2.pth"
        patchilizer_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=patch_length,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
        )
        self.patch_encoder = PatchEncoder(
            config=patchilizer_config,
            patch_size=patch_size,
            projection_layer=hidden_size,
        )
        self.model = CLaMP2Model(
            patch_encoder=self.patch_encoder, latent_dim=latent_dim
        )
        self.weights_url = weights_url
        self.patch_size = patch_size
        self.patch_length = patch_length
        try:
            self.checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        except Exception:
            self._download_checkpoint()
            self.checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )

    def _download_checkpoint(self) -> None:
        logger.info(
            f"Downloading CLaMP2 weights from: {self.weights_url} into {self.weights_path}"
        )
        download_file(self.weights_url, self.weights_path, verbose=True)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_list = []
        for i in range(0, len(data), self.patch_length):
            segment_list.append(data[i : i + self.patch_length])
        segment_list[-1] = data[-self.patch_length :]

        last_hidden_states_list = []

        for input_segment in segment_list:
            input_masks = torch.tensor([1] * input_segment.size(0))
            pad_indices = (
                torch.ones(
                    (self.patch_length - input_segment.size(0), self.patch_size)
                ).long()
                * self.patchilizer.pad_token_id
            )
            input_masks = torch.cat(
                (input_masks, torch.zeros(self.patch_length - input_segment.size(0))), 0
            )
            input_segment = torch.cat((input_segment, pad_indices), 0)
            last_hidden_states = self.model.get_music_features(
                music_inputs=input_segment.unsqueeze(0).to(self.device),
                music_masks=input_masks.unsqueeze(0).to(self.device),
            )
            last_hidden_states_list.append(last_hidden_states)

        full_chunk_cnt = len(data) // self.patch_length
        remain_chunk_len = len(data) % self.patch_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor(
                [self.patch_length] * full_chunk_cnt, device=self.device
            ).view(-1, 1)
        else:
            feature_weights = torch.tensor(
                [self.patch_length] * full_chunk_cnt + [remain_chunk_len],
                device=self.device,
            ).view(-1, 1)

        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = (
            last_hidden_states_list.sum(dim=0) / feature_weights.sum()
        )

        return last_hidden_states_list.unsqueeze(0).detach().cpu().numpy()


def encoder_mapper(encoder_name: str) -> Type[Encoder]:
    if encoder_name == "clamp2":
        return Clamp2Encoder
    # elif encoder_name == "simple_conv":
    #     return SimpleConvEncoder
    elif encoder_name == "dummy":
        return DummyEncoder
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}, possible: ['simple_conv']")
