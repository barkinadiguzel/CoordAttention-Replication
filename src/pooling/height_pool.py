import torch
import torch.nn as nn

class HeightPool(nn.Module):
    def forward(self, x):
        # x: [B, C, H, W] → [B, C, H, 1]
        return torch.mean(x, dim=3, keepdim=True)
