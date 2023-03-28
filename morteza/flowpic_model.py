import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F

MTU = 1514

def session_2d_histogram(ts, sizes, path, plot=False):
    # ts_norm = map(int, ((np.array(ts) - ts[0]) / (ts[-1] - ts[0])) * MTU)
    if len(ts) == 1:
        ts_norm = np.array([0.0])
    else:
        ts_norm = ((np.array(ts) - ts[0]) / (ts[-1] - ts[0])) * MTU
    H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(range(0, MTU + 1, 1), range(0, MTU + 1, 1)))

    if plot:
        plt.set_cmap('plasma')
        plt.pcolormesh(xedges, yedges, H)
        plt.xlim(0, MTU + 1)
        plt.ylim(0, MTU + 1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
    return torch.Tensor(H.astype(np.int32)).unsqueeze(0)

class TraceDataset(torch.utils.data.IterableDataset):
    def __init__(self, flow_times, flow_packets, flow_sizes, indexes, em):
        self.flow_times = flow_times
        self.flow_packets = flow_packets
        self.flow_sizes = flow_sizes
        self.indexes = indexes
        self.em = em
    
    def __iter__(self):
        for idx in self.indexes:
            c = 1 if self.flow_sizes[idx + 1] > self.em else 0
            yield session_2d_histogram(self.flow_times[idx], self.flow_packets[idx], '', False), c
    
    def __len__(self):
        return len(self.indexes)

class CNNModel(nn.Module):
    def __init__(self, filters=(10, 20)):
        super().__init__()
        self.filters = filters

        self.conv1 = nn.Conv2d(1, out_channels=self.filters[0], kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], out_channels=filters[1], kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.d2 = nn.Dropout(p=0.25)
        self.linear1 = nn.Linear(filters[1] * 93 * 93, 64)
        self.d3 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)

        x = self.d2(x)

        x = x.view(-1, self.filters[-1] * 93 * 93)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.d3(x)

        x = self.linear2(x)

        x = F.sigmoid(x)

        return x