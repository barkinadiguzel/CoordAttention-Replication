import torch.nn as nn

def relu():
    return nn.ReLU(inplace=True)

def sigmoid():
    return nn.Sigmoid()
