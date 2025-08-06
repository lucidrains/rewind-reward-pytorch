import torch
from torch.nn import Module

from x_transformers import Decoder

class CrossModalSequentialAggregator(Module):
    def __init__(self):
        super().__init__()
