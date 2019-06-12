'''Simple fully-connected models'''

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np


class DenseModel(nn.Module):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(400, 400)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
