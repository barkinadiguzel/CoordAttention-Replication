import torch.nn as nn
from backbone.mobilenetv2_blocks import InvertedResidual

class CoordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = InvertedResidual(32, 64)
        self.block2 = InvertedResidual(64, 128)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
