import torch
import torch.nn as nn
import torch.nn.functional as F

class MLFF(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[20],
                 func=F.leaky_relu, bn=False, output_size=1):
        """Multilayer Feed forward network

        Keyword Arguments:
            input_size {int} -- Input dimension (default: {2})
            hidden_sizes {list} -- Number of neurons in the hidden layers (default: {[20]})
            func {[type]} -- The activation function to use (default: {F.leaky_relu})
            bn {bool} -- Wether or not to use batch normalization (default: {False})
            output_size {int} -- Dimension of output
        """
        super(MLFF, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.hidden_sizes.insert(0, input_size)
        self.layers = nn.ModuleList([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(len(self.hidden_sizes)-1)])
        self.h2o = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.func = func
        self.bn = bn
        if self.bn:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in self.hidden_sizes[1:]])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.func(layer(x))
            if self.bn:
                x = self.batch_norms[i](x)
        x = self.h2o(x)
        return x

    def eval_seq(self, sequence: torch.Tensor):
        num_batch = sequence.shape[0]
        traj_len = sequence.shape[1]
        dim = sequence.shape[2]

        res = torch.zeros((num_batch, traj_len, self.output_size))
        res.to(sequence.device)
        for k in range(traj_len):
            res[:,k] = self.forward(sequence[:,k])

        return res