import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionNormalInjecter(nn.Module):
    """
    Fuse normal features into RGB features via lightweight cross-attention.

    Expects channel-first spatial feature maps from ViT: (B, C, H, W)
    Produces fused features with the same spatial shape as the RGB features.
    """

    def __init__(self, in_channels: int, out_channels: int = None, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # Project to token dim for attention
        self.q_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        # Multihead attention over flattened spatial tokens
        self.attn = nn.MultiheadAttention(self.out_channels, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

        # Output projection and residual
        self.out_proj = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(self.out_channels)

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

        # Simple residual with original rgb features (match channels if changed)
        if self.out_channels == c:
            fused = fused + rgb_feats

        return fused


