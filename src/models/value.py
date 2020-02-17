import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlff import MLFF


class FFValue(MLFF):
    def __init__(self, input_size=2, hidden_sizes=[20],
                 func=F.leaky_relu, bn=False):
        super(FFValue, self).__init__(input_size, hidden_sizes, func, bn)
