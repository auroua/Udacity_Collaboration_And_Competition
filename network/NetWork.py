import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False, seed=0):
        super(Network, self).__init__()
        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_in_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_out_dim)
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = F.relu(self.fc1(x))
            h1 = self.bn1(h1)
            h2 = F.relu(self.fc2(h1))
            # h2 = self.bn2(h2)
            h3 = torch.tanh(self.fc3(h2))
            return h3
        else:
            # critic network simply outputs a number
            h1 = F.leaky_relu(self.fc1(x))
            h1 = self.bn1(h1)
            h2 = F.leaky_relu(self.fc2(h1))
            # h2 = self.bn2(h2)
            h3 = F.leaky_relu(self.fc3(h2))
            return h3
