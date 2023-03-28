import torch
from torch import nn
from torch.functional import F


class CNNModel(nn.Module):
    def __init__(self, linear_size=4, filters=(30, 20, 10)):
        super().__init__()
        self.filters = filters
        self.linear_size = linear_size

        self.conv1 = nn.Conv1d(1, filters[0], 3)
        self.bn1 = nn.BatchNorm1d(filters[0])
        self.conv2 = nn.Conv1d(filters[0], filters[1], 3)
        self.bn2 = nn.BatchNorm1d(filters[1])
        self.conv3 = nn.Conv1d(filters[1], filters[2], 3)
        self.bn3 = nn.BatchNorm1d(filters[2])
        self.linear = nn.Linear(filters[2] * self.linear_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.bn2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.bn3(x)
        
        x = x.view(-1, self.filters[-1] * self.linear_size)

        x = self.linear(x)
        x = F.sigmoid(x)

        return x