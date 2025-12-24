"""
SCTuner.py - Parameter Efficient Fine-Tuning Module for MedSegDiff
This module implements the Skip Connection Editing (SCEdit) tuner with FiLM modulation.
"""

from typing import List
import torch
import torch.nn as nn
# 假設你的專案結構中 guided_diffusion/nn.py 包含 conv_nd
# 如果報錯找不到 conv_nd，請確認你的 nn.py 路徑
from .nn import conv_nd 

class CSCTuner(nn.Module):
    """
    Context-Aware Skip Connection Tuner (CSC-Tuner).
    
    Mechanism:
    1. PreHint: Compresses the input context (image/mask) into a global embedding vector.
    2. Stage Heads: Generates scale & bias parameters for each U-Net encoder stage.
    3. Modulation: Applies FiLM (h' = h * (1 + scale) + bias) to feature maps.
    
    Key Feature:
    - Zero Initialization: The final projection layers are initialized to zero.
      This ensures that at the start of training, scale=0 and bias=0, 
      so the model behaves exactly like the pre-trained backbone (h' = h).
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

        # 1. Pre-hint Network: Extracts global context features
        # Structure: Conv -> SiLU -> Downsample -> ... -> Global Average Pool
        ch = embed_channels
        self.pre_hint = nn.Sequential(
            conv_nd(dims, pre_hint_in_channels, ch // 2, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, ch // 2, ch // 2, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, ch // 2, ch, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        # Global Average Pooling (Adaptive ensures output is 1x1 or 1x1x1)
        if dims == 3:
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        # 2. Stage Heads: One MLP per encoder resolution level
        self.heads = nn.ModuleList()
        
        for c_out in self.input_block_channels:
            # Each head predicts 2 * c_out (scale + bias)
            head = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.SiLU(),
                nn.Linear(embed_channels, 2 * c_out),
            )
            
            # ============================================================
            # [CRITICAL] Zero Initialization
            # 這是讓你的 DICE 分數不會從 0.4 開始的關鍵
            # 確保初始狀態下 Tuner 不對骨幹網路產生任何影響
            # ============================================================
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            
            self.heads.append(head)

    def _embed_context(self, c: torch.Tensor) -> torch.Tensor:
        """Computes the global embedding vector from the context input."""
        # c shape: [B, C_in, H, W] or [B, C_in, D, H, W]
        feat = self.pre_hint(c)      # Extract spatial features
        feat = self.gap(feat)        # Global Average Pooling -> [B, C_embed, 1, 1...]
        feat = feat.flatten(1)       # Flatten to [B, C_embed]
        return feat

    def forward(self, c: torch.Tensor, h: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        Apply modulation to the feature map `h`.
        
        Args:
            c: Context tensor (e.g., input image without noise), shape [B, C_in, ...]
            h: Feature map from U-Net encoder, shape [B, C_feat, H, W]
            stage_idx: Index of the current encoder block to select the correct head.
            
        Returns:
            Modulated feature map with same shape as h.
        """
        if not self.heads:
            return h
            
        # Safety check for index
        stage_idx = max(0, min(stage_idx, len(self.heads) - 1))

        # 1. Get Global Context Embedding
        # (Ideally, compute this once outside and pass it in to save compute, 
        # but computing here is safer for "drop-in" compatibility)
        vec = self._embed_context(c)  # [B, Embed_Dim]

        # 2. Predict Scale and Bias
        pred = self.heads[stage_idx](vec)  # [B, 2 * C_feat]
        
        # 3. Split into Scale and Bias
        ch = h.shape[1]
        scale, bias = pred.split([ch, ch], dim=1)

        # 4. View for Broadcasting
        # Reshape [B, C] -> [B, C, 1, 1] (2D) or [B, C, 1, 1, 1] (3D)
        # This allows element-wise multiplication with spatial feature map h
        view_shape = [h.shape[0], ch] + [1] * (h.dim() - 2)
        scale = scale.view(*view_shape)
        bias = bias.view(*view_shape)

        # 5. Apply FiLM Modulation
        # Formula: h' = h * (1 + scale) + bias
        # Initial state: scale=0, bias=0  => h' = h * 1 + 0 = h (Identity)
        return h * (1.0 + scale) + bias