import torch
import torch.nn as nn

class WidthPool(nn.Module):
    def forward(self, x):
        # x: [B, C, H, W] → [B, C, 1, W]
        return torch.mean(x, dim=2, keepdim=True)
