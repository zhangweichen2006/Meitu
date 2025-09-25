import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionNormalInjecter(nn.Module):
    """
    Fuse normal features into RGB features via lightweight cross-attention.

    Expects channel-first spatial feature maps from ViT: (B, C, H, W)
    Produces fused features with the same spatial shape as the RGB features.
    """

    def __init__(self, in_channels: int, out_channels: int = None, num_heads: int = 8, dropout: float = 0.0, alpha: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        # Project to token dim for attention
        self.q_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        # Multihead attention over flattened spatial tokens
        self.attn = nn.MultiheadAttention(self.out_channels, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

        # Output projection and residual
        self.out_proj = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(self.out_channels)
        # If using a bottleneck (out_channels != in_channels), project back to in_channels
        self.final_proj = None
        if self.out_channels != self.in_channels:
            self.final_proj = nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1)

    def forward(self, rgb_feats: torch.Tensor, normal_feats: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feats.shape

        # 1x1 conv projections
        q = self.q_proj(rgb_feats)
        k = self.k_proj(normal_feats)
        v = self.v_proj(normal_feats)

        # (B, C, H, W) -> (B, HW, C)
        q_tokens = q.flatten(2).transpose(1, 2)
        k_tokens = k.flatten(2).transpose(1, 2)
        v_tokens = v.flatten(2).transpose(1, 2)

        # LayerNorm on channel dim for token representations
        q_tokens = self.norm(q_tokens)
        k_tokens = self.norm(k_tokens)
        v_tokens = self.norm(v_tokens)

        # Cross-attention: query from RGB, key/value from normal
        fused_tokens, _ = self.attn(q_tokens, k_tokens, v_tokens, need_weights=False)

        # Back to spatial and residual add
        fused = fused_tokens.transpose(1, 2).reshape(b, self.out_channels, h, w)
        fused = self.out_proj(fused)
        if self.final_proj is not None:
            fused = self.final_proj(fused)

        # Blend with original RGB features
        if self.alpha >= 1.0:
            return fused
        if self.alpha <= 0.0:
            return rgb_feats
        return self.alpha * fused + (1.0 - self.alpha) * rgb_feats


class FullyConnectedNormalInjecter(nn.Module):
    """
    Fuse by per-pixel MLP over concatenated RGB and normal features.
    Input/Output: (B, C, H, W). If out_channels != in_channels, project back.
    """

    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, dropout: float = 0.0, alpha: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_channels = hidden_channels or max(in_channels // 2, 64)
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1),
        )
        self.final_proj = None
        if self.out_channels != self.in_channels:
            self.final_proj = nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1)

    def forward(self, rgb_feats: torch.Tensor, normal_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb_feats, normal_feats], dim=1)
        y = self.mlp(x)
        if self.final_proj is not None:
            y = self.final_proj(y)
        if self.alpha >= 1.0:
            return y
        if self.alpha <= 0.0:
            return rgb_feats
        return self.alpha * y + (1.0 - self.alpha) * rgb_feats


class AdditionNormalInjecter(nn.Module):
    """
    Simple weighted addition fusion: alpha * normal + (1 - alpha) * rgb.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, rgb_feats: torch.Tensor, normal_feats: torch.Tensor) -> torch.Tensor:
        return self.alpha * normal_feats + (1.0 - self.alpha) * rgb_feats


