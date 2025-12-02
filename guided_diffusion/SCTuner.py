from typing import List
import torch
import torch.nn as nn
from .nn import conv_nd


class CSCTunerSimple(nn.Module):
    """
    A simple, self-contained context-aware tuner for this repo.
    - PreHint: small CNN over the input context (input without last channel)
      to produce a global embedding with global average pooling.
    - For each encoder stage, predict (scale, bias) vectors and modulate the
      feature map h: h' = h * (1 + scale) + bias.

    Instantiate with the list of encoder stage channel sizes, then call from
    the UNet forward as: h = tuner(c, h, stage_idx).
    """

    def __init__(
        self,
        input_block_channels: List[int],
        pre_hint_in_channels: int,
        embed_channels: int = 128,
        dims: int = 2,
    ) -> None:
        super().__init__()
        self.input_block_channels = list(input_block_channels)
        self.embed_channels = embed_channels
        self.dims = dims

        # Pre-hint embedding network: a few convs with downsampling, then GAP.
        ch = embed_channels
        self.pre_hint = nn.Sequential(
            conv_nd(dims, pre_hint_in_channels, ch // 2, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, ch // 2, ch // 2, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, ch // 2, ch, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        if dims == 3:
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        # Per-stage heads: Linear(embed) -> Linear(2*C) where C is stage channels.
        self.heads = nn.ModuleList()
        for c_out in self.input_block_channels:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels),
                    nn.SiLU(),
                    nn.Linear(embed_channels, 2 * c_out),
                )
            )

    def _embed_context(self, c: torch.Tensor) -> torch.Tensor:
        # c: [N, Cc, H, W] or 3D -> [N, D]
        e = self.pre_hint(c)
        e = self.gap(e).flatten(1)
        return e

    def forward(self, c: torch.Tensor, h: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        c: context tensor (input without the last channel), shape [N, Cc, ...]
        h: feature map to modulate, shape [N, Ch, H, W]
        stage_idx: index of the stage head to use
        returns: modulated feature map, same shape as h
        """
        if not self.heads:
            return h
        stage_idx = max(0, min(stage_idx, len(self.heads) - 1))
        vec = self._embed_context(c)  # [N, D]
        pred = self.heads[stage_idx](vec)  # [N, 2*Ch]
        ch = h.shape[1]
        scale, bias = pred.split([ch, ch], dim=1)
        # reshape for broadcasting over spatial dims
        view_shape = [h.shape[0], ch] + [1] * (h.dim() - 2)
        scale = scale.view(*view_shape)
        bias = bias.view(*view_shape)
        return h * (1.0 + scale) + bias
