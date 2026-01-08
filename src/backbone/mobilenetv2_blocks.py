import torch.nn as nn
from attention.coord_block import CoordAttentionBlock

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, use_att=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.att = CoordAttentionBlock(out_ch) if use_att else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.att(x)
        return x
