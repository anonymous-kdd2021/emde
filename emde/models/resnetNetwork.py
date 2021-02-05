import torch.nn as nn
import torch.nn.functional as F

"""
3 layer feed forward netowrk with residual connections and batch normalization.
"""

class ResNetModel(nn.Module):
    def __init__(self, n_sketches, sketch_dim, input_dim, output_dim, hidden_size):
        """
        :param int n_sketches: sketch depth
        :param int sketch_dim: sketch width
        :param int input_dim: input dimension to feed forward network
        :param int output_dim: output dimension of feed forward network
        :param int hidden_size: hidden size of feed forward network
        """
        super().__init__()
        self.output_dim = output_dim
        self.n_sketches = n_sketches
        self.sketch_dim = sketch_dim
        self.projection = nn.Linear(input_dim, hidden_size)
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, self.output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x_input):
        """
        Feed forward network with residual connections.
        """
        x_proj = self.projection(x_input)
        x = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x) + x_proj))
        x = self.l_output(self.bn3(F.leaky_relu(self.l3(x) + x_proj)))
        x = F.softmax(x.view(-1, self.sketch_dim, self.n_sketches), dim=1).view(-1, self.output_dim)
        return x