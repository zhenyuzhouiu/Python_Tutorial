import torch
from torch.nn import functional as F


class TensorVstackGradient(torch.nn.Module):
    def __init__(self):
        super(TensorVstackGradient, self).__init__()

    def forward(self):