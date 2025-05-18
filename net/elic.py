# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## This module contains modules implementing standard synthesis and analysis transforms

from typing import List, Optional

import torch
import torch.nn as nn
from compressai.layers import GDN, AttentionBlock
from torch import Tensor

from layer.layer_utils import make_conv, make_deconv


class ConvGDNAnalysis(nn.Module):
    def __init__(
            self, network_channels: int = 128, compression_channels: int = 192
    ) -> None:
        """
        Analysis transfrom from scale hyperprior (https://arxiv.org/abs/1802.01436)

        Encodes each image in a video independently.
        """
        super().__init__()
        self._compression_channels = compression_channels

        self.transforms = nn.Sequential(
            make_conv(3, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, compression_channels, kernel_size=5, stride=2),
        )
        self._num_down = 4

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    @property
    def num_downsampling_layers(self) -> int:
        return self._num_down

    def forward(self, video_frames: Tensor) -> Tensor:
        """
        Args:
            video_frames: frames of a batch of clips. Expected shape [B, T, C, H, W],
                which is reshaped to [BxT, C, H, W], hyperprior model encoder is applied
                and output is reshaped back to [B, T, <compression_channels>, h, w].
        Returns:
            embeddings: embeddings of shape [B, T, <compression_channels>, h, w], obtained
                by running ScaleHyperprior.image_analysis().
        """
        assert (
                video_frames.dim() == 5
        ), f"Expected [B, T, C, H, W] got {video_frames.shape}"
        embeddings = self.transforms(video_frames.reshape(-1, *video_frames.shape[2:]))
        return embeddings.reshape(*video_frames.shape[:2], *embeddings.shape[1:])


class ConvGDNSynthesis(nn.Module):
    def __init__(
            self, network_channels: int = 128, compression_channels: int = 192
    ) -> None:
        """
        Synthesis transfrom from scale hyperprior (https://arxiv.org/abs/1802.01436)

        Decodes each image in a video independently
        """

        super().__init__()
        self._compression_channels = 192

        self.transforms = nn.Sequential(
            make_deconv(
                compression_channels, network_channels, kernel_size=5, stride=2
            ),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, 3, kernel_size=5, stride=2),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        """
        Args:
            x: the (reconstructed) latent embdeddings to be decoded to images,
                expected shape [B, T, C, H, W]
            frames_shape: shape of the video clip to be reconstructed.
        Returns:
            reconstruction: reconstruction of the original video clip with shape
                [B, T, C, H, W] = frames_shape.
        """
        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        # Treat T as part of the Batch dimension, storing values to reshape back
        B, T, *_ = x.shape
        x = x.reshape(-1, *x.shape[2:])
        assert len(frames_shape) == 5

        x = self.transforms(x)  # final reconstruction
        x = x.reshape(B, T, *x.shape[1:])
        return x[..., : frames_shape[-2], : frames_shape[-1]]


class ResidualUnit(nn.Module):
    """Simple residual unit"""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            make_conv(N, N // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N // 2, kernel_size=3),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N, kernel_size=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out += identity
        out = self.activation(out)
        return out


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim // 2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class AdaLN(nn.Module):
    default_embedding_dim = 256

    def __init__(self, dim, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2 * dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2 * dim))
        )

    def forward(self, x, emb):
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConditionalResidualUnit(nn.Module):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.pre_conv = make_conv(N, N // 2, kernel_size=1)
        self.adaLN = AdaLN(N // 2)
        self.post_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            make_conv(N // 2, N // 2, kernel_size=3),
            # nn.ReLU(inplace=True),
            make_conv(N // 2, N, kernel_size=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        identity = x
        out = self.pre_conv(x)
        out = self.adaLN(out, emb)
        out = self.post_conv(out)
        out += identity
        out = self.activation(out)
        return out


class ELICAnalysis(nn.Module):
    def __init__(
            self,
            num_residual_blocks=3,
            channels: List[int] = [256, 256, 256, 192],
            compression_channels: Optional[int] = None,
            max_frames: Optional[int] = None,
    ) -> None:
        """Analysis transform from ELIC (https://arxiv.org/abs/2203.10886), which
        can be configured to match the one from "Devil's in the Details"
        (https://arxiv.org/abs/2203.08450).

        Args:
            num_residual_blocks: defaults to 3.
            channels: defaults to [128, 160, 192, 192].
            compression_channels: optional, defaults to None. If provided, it must equal
                the last element of `channels`.
            max_frames: optional, defaults to None. If provided, the input is chunked
                into max_frames elements, otherwise the entire batch is processed at
                once. This is useful when large sequences are to be processed and can
                be used to manage memory a bit better.
        """
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {len(channels)}).")
        if compression_channels is not None and compression_channels != channels[-1]:
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{compression_channels} vs. {channels}"
            )
        self._compression_channels = (
            compression_channels if compression_channels is not None else channels[-1]
        )
        self._max_frames = max_frames

        def res_units(N):
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [3] + channels

        self.transforms = nn.Sequential(
            make_conv(channels[0], channels[1], kernel_size=5, stride=2),
            *res_units(channels[1]),
            make_conv(channels[1], channels[2], kernel_size=5, stride=2),
            *res_units(channels[2]),
            AttentionBlock(channels[2]),
            make_conv(channels[2], channels[3], kernel_size=5, stride=2),
            *res_units(channels[3]),
            make_conv(channels[3], channels[4], kernel_size=5, stride=2),
            AttentionBlock(channels[4]),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.transforms(x)
        return x


class ELICSynthesis(nn.Module):
    def __init__(
            self,
            num_residual_blocks=3,
            channels: List[int] = [192, 256, 256, 3],
            output_channels: Optional[int] = None
    ) -> None:
        """
        Synthesis transform from ELIC (https://arxiv.org/abs/2203.10886).

        Args:
            num_residual_blocks: defaults to 3.
            channels: _defaults to [192, 160, 128, 3].
            output_channels: optional, defaults to None. If provided, it must equal
                the last element of `channels`.
        """
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
        if output_channels is not None and output_channels != channels[-1]:
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{output_channels} vs. {channels}"
            )

        self._compression_channels = channels[0]

        def res_units(N: int) -> List:
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [channels[0]] + channels
        self.transforms = nn.Sequential(
            AttentionBlock(channels[0]),
            make_deconv(channels[0], out_channels=channels[1], kernel_size=5, stride=2),
            *res_units(channels[1]),
            make_deconv(channels[1], out_channels=channels[2], kernel_size=5, stride=2),
            AttentionBlock(channels[2]),
            *res_units(channels[2]),
            make_deconv(channels[2], out_channels=channels[3], kernel_size=5, stride=2),
            *res_units(channels[3]),
            make_deconv(channels[3], out_channels=channels[4], kernel_size=5, stride=2),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.transforms(x)
        return x


class ConditionalAttentionBlock(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.conv_a = nn.Sequential(ConditionalResidualUnit(N),
                                    ConditionalResidualUnit(N),
                                    ConditionalResidualUnit(N))

        self.conv_b1 = nn.Sequential(
            ConditionalResidualUnit(N),
            ConditionalResidualUnit(N),
            ConditionalResidualUnit(N),
        )
        self.conv_b2 = make_conv(N, N, kernel_size=1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        identity = x
        a = x
        for conv_a in self.conv_a:
            a = conv_a(a, emb)
        for conv_b1 in self.conv_b1:
            x = conv_b1(x, emb)
        b = self.conv_b2(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class AdaptiveELICSynthesis(nn.Module):
    def __init__(
            self,
            num_residual_blocks=3,
            channels: List[int] = [192, 256, 256, 3],
            output_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
        if output_channels is not None and output_channels != channels[-1]:
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{output_channels} vs. {channels}"
            )

        self._compression_channels = channels[0]

        def cond_res_units(N: int) -> List:
            return [ConditionalResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [channels[0]] + channels
        self.transforms_1 = ConditionalAttentionBlock(channels[0])
        self.deconv_1 = make_deconv(channels[0], out_channels=channels[1], kernel_size=5, stride=2)
        self.transforms_2 = nn.Sequential(*cond_res_units(channels[1]))
        self.deconv_2 = make_deconv(channels[1], out_channels=channels[2], kernel_size=5, stride=2)
        self.transforms_3 = nn.Sequential(ConditionalAttentionBlock(channels[2]),
                                          *cond_res_units(channels[2]))
        self.deconv_3 = make_deconv(channels[2], out_channels=channels[3], kernel_size=5, stride=2)
        self.transforms_4 = nn.Sequential(*cond_res_units(channels[3]))
        self.deconv_4 = make_deconv(channels[3], out_channels=channels[4], kernel_size=5, stride=2)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.transforms_1(x, emb)
        x = self.deconv_1(x)
        for transforms_2 in self.transforms_2:
            x = transforms_2(x, emb)
        x = self.deconv_2(x)
        for transforms_3 in self.transforms_3:
            x = transforms_3(x, emb)
        x = self.deconv_3(x)
        for transforms_4 in self.transforms_4:
            x = transforms_4(x, emb)
        x = self.deconv_4(x)
        return x


if __name__ == '__main__':
    g_s = AdaptiveELICSynthesis()
    y = torch.randn(1, 192, 8, 8)
    emb = torch.randn(1, 256)
    x = g_s(y, emb)
    print(x.shape)
