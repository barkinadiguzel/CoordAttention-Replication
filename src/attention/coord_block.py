import torch
import torch.nn as nn
from pooling.height_pool import HeightPool
from pooling.width_pool import WidthPool
from pooling.concat_pool import ConcatPool
from layers.conv_layer import Conv1x1
from layers.activation import relu
from attention.coord_attention import CoordAttentionGen

class CoordAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.h_pool = HeightPool()
        self.w_pool = WidthPool()
        self.concat = ConcatPool()

        self.conv1 = Conv1x1(channels, channels // reduction)
        self.relu = relu()

        self.attn_gen = CoordAttentionGen(channels, reduction)

    def forward(self, x):
        B, C, H, W = x.shape

        # Eq(4)-(5)
        z_h = self.h_pool(x)
        z_w = self.w_pool(x)

        # concat
        f = self.concat(z_h, z_w)

        # Eq(6)
        f = self.relu(self.conv1(f))

        # Eq(7)-(8)
        g_h, g_w = self.attn_gen(f, H, W)

        # Eq(9)
        out = x * g_h * g_w
        return out
