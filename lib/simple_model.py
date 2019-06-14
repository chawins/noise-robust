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


class DenseModelV2(nn.Module):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModelV2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 400)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x
