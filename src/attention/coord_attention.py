import torch
import torch.nn as nn
from layers.conv_layer import Conv1x1
from layers.activation import relu, sigmoid

class CoordAttentionGen(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        mid = channels // reduction

        self.conv1 = Conv1x1(channels, mid)
        self.relu = relu()

        self.conv_h = Conv1x1(mid, channels)
        self.conv_w = Conv1x1(mid, channels)

        self.sigmoid = sigmoid()

    def forward(self, f, H, W):
        f_h, f_w = torch.split(f, [H, W], dim=2)

        g_h = self.sigmoid(self.conv_h(f_h))  # [B,C,H,1]
        g_w = self.sigmoid(self.conv_w(f_w))  # [B,C,W,1]

        g_w = g_w.permute(0,1,3,2)  # [B,C,1,W]
        return g_h, g_w
