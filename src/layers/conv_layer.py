import torch.nn as nn

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)
