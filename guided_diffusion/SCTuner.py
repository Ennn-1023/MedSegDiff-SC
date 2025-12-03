from typing import List
import torch
import torch.nn as nn
from .nn import conv_nd


def conv_nd(dims, *args, **kwargs):
    """根據維度選擇卷積層"""
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """將模組的權重和偏置初始化為零"""
    for p in module.parameters():
        p.detach().zero_()
    return module


class SCEAdapter(nn.Module):
    """輕量級適配器：2層MLP + 殘差連接"""

    def __init__(
            self,
            dim: int,
            adapter_length: int,
            zero_init_last: bool = True,
            use_bias: bool = True,
            act_layer=nn.GELU,
    ):
        super().__init__()
        self.ln1 = nn.Linear(dim, adapter_length, bias=use_bias)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim, bias=use_bias)

        # 初始化
        nn.init.kaiming_uniform_(self.ln1.weight)
        if zero_init_last:
            nn.init.zeros_(self.ln2.weight)
            if use_bias:
                nn.init.zeros_(self.ln2.bias)
        else:
            nn.init.kaiming_uniform_(self.ln2.weight)

    def forward(self, x: torch.Tensor, x_shortcut: torch.Tensor = None, use_shortcut: bool = True):
        if x_shortcut is None:
            x_shortcut = x

        x_shape = x.shape
        # 4D特徵圖需展平處理
        if len(x_shape) == 4:
            b, d, h, w = x_shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, d)

        out = self.ln2(self.activate(self.ln1(x)))

        # 還原形狀
        if len(x_shape) == 4:
            out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        if use_shortcut:
            out = x_shortcut + out
        return out


class SCTuner(nn.Module):
    """單層 Skip Connection Tuner"""

    def __init__(self, dim: int, tuner_length: int, **kwargs):
        super().__init__()
        self.tuner_op = SCEAdapter(dim=dim, adapter_length=tuner_length)

    def forward(self, x, x_shortcut=None, use_shortcut=True):
        return self.tuner_op(x, x_shortcut, use_shortcut)


class CSCTuner(nn.Module):
    """
    完整的 Context-aware Skip Connection Tuner
    包含：
    1. PreHint: 從控制輸入提取全域特徵
    2. DenseHint: 逐層空間卷積路徑（用於 encoder）
    3. LSCTuner: 每層的 skip connection 適配器（用於 decoder）
    """

    class CSCTuner(nn.Module):
        def __init__(
                self,
                input_block_channels: List[int],
                output_block_channels: List[int],  # 新增：明确 decoder 通道配置
                input_down_flag: List[bool],
                pre_hint_in_channels: int = 3,
                pre_hint_out_channels: int = 256,
                pre_hint_dim_ratio: float = 1.0,
                dense_hint_kernel: int = 3,
                down_ratio: float = 1.0,
                dims: int = 2,
        ):
            super().__init__()
            self.scale = 1.0

            # ==================== 1. PreHint ====================
            ch = pre_hint_out_channels
            self.pre_hint_blocks = nn.Sequential(
                conv_nd(dims, pre_hint_in_channels, int(16 * pre_hint_dim_ratio), 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, int(16 * pre_hint_dim_ratio), int(16 * pre_hint_dim_ratio), 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, int(16 * pre_hint_dim_ratio), int(32 * pre_hint_dim_ratio), 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, int(32 * pre_hint_dim_ratio), int(32 * pre_hint_dim_ratio), 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, int(32 * pre_hint_dim_ratio), int(96 * pre_hint_dim_ratio), 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, int(96 * pre_hint_dim_ratio), int(96 * pre_hint_dim_ratio), 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, int(96 * pre_hint_dim_ratio), ch, 3, padding=1, stride=2),
            )

            # ==================== 2. DenseHint（Encoder 注入）====================
            self.dense_hint_blocks = nn.ModuleList()
            stride_list = [2 if flag else 1 for flag in input_down_flag]

            for i, chan in enumerate(input_block_channels):
                padding = 1 if dense_hint_kernel == 3 else 0
                self.dense_hint_blocks.append(
                    nn.Sequential(
                        nn.SiLU(),
                        zero_module(
                            conv_nd(dims, ch, chan, dense_hint_kernel,
                                    padding=padding, stride=stride_list[i])
                        )
                    )
                )
                # ✅ 修正：考虑下采样时的通道变化
                ch = chan if stride_list[i] == 1 else chan * 2

            # ==================== 3. LSCTuner（Decoder Skip）====================
            self.lsc_tuner_blocks = nn.ModuleList()
            for chan in output_block_channels:  # ✅ 使用 decoder 通道配置
                tuner_length = int(chan * down_ratio)
                self.lsc_tuner_blocks.append(
                    SCTuner(dim=chan, tuner_length=tuner_length)
                )

            # ✅ 新增：Identity 用于参考代码的 residual 判定
            self.lsc_identity = nn.ModuleList([nn.Identity() for _ in output_block_channels])
    def forward_dense_hint(self, hint: torch.Tensor) -> List[torch.Tensor]:
        """
        處理 DenseHint 路徑（用於 UNet encoder）
        Args:
            hint: [B, C, H, W] 控制輸入
        Returns:
            List of dense hints for each encoder stage
        """
        h = self.pre_hint_blocks(hint)
        dense_hints = []
        for block in self.dense_hint_blocks:
            h = block(h)
            dense_hints.append(h)
        return dense_hints

    def forward_lsc_tuner(self, x: torch.Tensor, stage_idx: int,
                          x_shortcut: torch.Tensor = None) -> torch.Tensor:
        """
        應用 LSCTuner（用於 UNet decoder 的 skip connection）
        Args:
            x: decoder 特徵
            stage_idx: 層索引（對應 output_blocks）
            x_shortcut: encoder 傳來的 skip connection
        Returns:
            調整後的特徵
        """
        if stage_idx < 0 or stage_idx >= len(self.lsc_tuner_blocks):
            return x
        return self.lsc_tuner_blocks[stage_idx](x, x_shortcut, use_shortcut=True)


# ==================== 使用範例 ====================
if __name__ == "__main__":
    # 假設 UNet 的配置
    input_block_channels = [320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280]
    input_down_flag = [False, False, True, False, False, True, False, False, True, False, False, False]

    tuner = CSCTuner(
        input_block_channels=input_block_channels,
        input_down_flag=input_down_flag,
        pre_hint_in_channels=3,
        pre_hint_out_channels=256,
        dense_hint_kernel=3,
        down_ratio=1.0,
    )

    # 模擬控制輸入
    hint = torch.randn(2, 3, 512, 512)

    # 1. Encoder 階段：取得 dense hints
    dense_hints = tuner.forward_dense_hint(hint)
    print(f"Dense hints count: {len(dense_hints)}")

    # 2. Decoder 階段：應用 LSCTuner
    decoder_feature = torch.randn(2, 320, 64, 64)
    skip_connection = torch.randn(2, 320, 64, 64)
    output = tuner.forward_lsc_tuner(decoder_feature, stage_idx=0, x_shortcut=skip_connection)
    print(f"LSCTuner output shape: {output.shape}")

# class CSCTuner(nn.Module):
#     """
#     A simple, self-contained context-aware tuner for this repo.
#     - PreHint: small CNN over the input context (input without last channel)
#       to produce a global embedding with global average pooling.
#     - For each encoder stage, predict (scale, bias) vectors and modulate the
#       feature map h: h' = h * (1 + scale) + bias.
#
#     Instantiate with the list of encoder stage channel sizes, then call from
#     the UNet forward as: h = tuner(c, h, stage_idx).
#     """
#
#     def __init__(
#         self,
#         input_block_channels: List[int],
#         pre_hint_in_channels: int,
#         embed_channels: int = 128,
#         dims: int = 2,
#     ) -> None:
#         super().__init__()
#         self.input_block_channels = list(input_block_channels)
#         self.embed_channels = embed_channels
#         self.dims = dims
#
#         # Pre-hint embedding network: a few convs with downsampling, then GAP.
#         ch = embed_channels
#         self.pre_hint = nn.Sequential(
#             conv_nd(dims, pre_hint_in_channels, ch // 2, 3, padding=1),
#             nn.SiLU(),
#             conv_nd(dims, ch // 2, ch // 2, 3, padding=1, stride=2),
#             nn.SiLU(),
#             conv_nd(dims, ch // 2, ch, 3, padding=1, stride=2),
#             nn.SiLU(),
#         )
#         if dims == 3:
#             self.gap = nn.AdaptiveAvgPool3d(1)
#         else:
#             self.gap = nn.AdaptiveAvgPool2d(1)
#
#         # Per-stage heads: Linear(embed) -> Linear(2*C) where C is stage channels.
#         self.heads = nn.ModuleList()
#         for c_out in self.input_block_channels:
#             self.heads.append(
#                 nn.Sequential(
#                     nn.Linear(embed_channels, embed_channels),
#                     nn.SiLU(),
#                     nn.Linear(embed_channels, 2 * c_out),
#                 )
#             )
#
#     def _embed_context(self, c: torch.Tensor) -> torch.Tensor:
#         # c: [N, Cc, H, W] or 3D -> [N, D]
#         e = self.pre_hint(c)
#         e = self.gap(e).flatten(1)
#         return e
#
#     def forward(self, c: torch.Tensor, h: torch.Tensor, stage_idx: int) -> torch.Tensor:
#         """
#         c: context tensor (input without the last channel), shape [N, Cc, ...]
#         h: feature map to modulate, shape [N, Ch, H, W]
#         stage_idx: index of the stage head to use
#         returns: modulated feature map, same shape as h
#         """
#         if not self.heads:
#             return h
#         stage_idx = max(0, min(stage_idx, len(self.heads) - 1))
#         vec = self._embed_context(c)  # [N, D]
#         pred = self.heads[stage_idx](vec)  # [N, 2*Ch]
#         ch = h.shape[1]
#         scale, bias = pred.split([ch, ch], dim=1)
#         # reshape for broadcasting over spatial dims
#         view_shape = [h.shape[0], ch] + [1] * (h.dim() - 2)
#         scale = scale.view(*view_shape)
#         bias = bias.view(*view_shape)
#         return h * (1.0 + scale) + bias
