import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlff import MLFF


class FFPolicy(MLFF):
    def __init__(self, input_size=2, hidden_sizes=[20],
                 func=F.leaky_relu, bn=False, actions=2):
        super(FFPolicy, self).__init__(input_size, hidden_sizes, func, bn,
                                       actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = super(FFPolicy, self).forward(x)

        return self.softmax(x)
