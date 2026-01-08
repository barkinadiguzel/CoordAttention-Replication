import torch.nn as nn

def batch_norm(channels):
    return nn.BatchNorm2d(channels)
