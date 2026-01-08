import torch
import torch.nn as nn

class ConcatPool(nn.Module):
    def forward(self, z_h, z_w):
        # z_h: [B, C, H, 1]
        # z_w: [B, C, 1, W] → permute to [B, C, W, 1]
        z_w = z_w.permute(0, 1, 3, 2)
        return torch.cat([z_h, z_w], dim=2)  # [B, C, H+W, 1]
