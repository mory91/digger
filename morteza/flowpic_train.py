import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import preprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F

from tqdm import tqdm

import random

from flowpic_model import *

from trace_process import *

TIME_DELTA = 500 * NANO_TO_MICRO
BATCH_SIZE = 32
N_EPOCHS = 5


def get_packets(file_path):
    df = pd.read_csv(
        file_path, 
        sep='\t', 
        lineterminator='\n', 
        header=None,
        index_col=False,
        names=['timestamp', 'size', 'src', 'dest', 'dir'], 
        dtype={'size': "int16", 'src': "category", 'dest': "category", "timestamp": "int64", "size": "int64", "dir": "int8"},
    )
    df = df[df['dir'] == 1][["timestamp", "size"]]
    df = df.sort_values(by='timestamp')
    return df.values

def get_train_test_indexes(total_size, split_factor):
    TRAIN_SIZE = 20_000#int(split_factor * total_size)
    TEST_SIZE = 5_000#int((1 - split_factor) * total_size)
    indexes = list(range(total_size - 1))
    random.shuffle(indexes)
    train_indexes = indexes[:TRAIN_SIZE]
    test_indexes = indexes[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    return train_indexes, test_indexes

def get_loss_and_correct(model, batch, criterion, device):
        data, target = batch
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        output = model(data)
        output = torch.squeeze(output)

        loss = criterion(output, target)

        pred = torch.round(output).int()
        true_num = pred.eq(target.int().data.view_as(pred)).sum()

        return loss, true_num

def step(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    trace = get_packets("../data/11/node-2/train/packets")
    for time_d in range(500, 5000, 250):
        time_delta = time_d * NANO_TO_MICRO
        flow_packets, flow_times, flow_sizes, flows_span = get_flow_trace_time(trace, time_delta)
        EM = np.median(flow_sizes)
        print("Number of flows: ", len(flow_sizes))
        print("EM: ", EM)

        TRAIN_TEST_SPLIT = 0.8
        TOTAL_SIZE = len(flow_sizes)
        train_indexes, test_indexes = get_train_test_indexes(TOTAL_SIZE, TRAIN_TEST_SPLIT)

        train_dataset = TraceDataset(flow_times, flow_packets, flow_sizes, train_indexes, EM)
        test_dataset = TraceDataset(flow_times, flow_packets, flow_sizes, test_indexes, EM)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        criterion = nn.BCELoss()
        model = CNNModel()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.5)

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        train_losses = []
        train_accuracies = []

        pbar = tqdm(range(N_EPOCHS))

        for i in pbar:
            total_train_loss = 0.0
            total_train_correct = 0.0

            model.train()

            for batch in tqdm(train_dataloader, leave=False):
                loss, correct = get_loss_and_correct(model, batch, criterion, device)
                step(loss, optimizer)
                total_train_loss += loss.item()
                total_train_correct += correct.item()

            mean_train_loss = total_train_loss / len(train_dataset)
            train_accuracy = total_train_correct / len(train_dataset)

            train_losses.append(mean_train_loss)

            train_accuracies.append(train_accuracy)

            pbar.set_postfix({'train_loss': mean_train_loss, 'train_accuracy': train_accuracy})

            print("train_accuarcy: ", train_accuracy, time_d)



        torch.save(model.state_dict(), "model.pt")

        cpu = torch.device('cpu')
        cpu_model = CNNModel()
        cpu_model.load_state_dict(torch.load("model.pt", map_location=cpu))


        total = 0
        correct = 0

        for batch in tqdm(test_dataloader, leave=False):
            model.eval()
            with torch.no_grad():
                data, target = batch
                data, target = data.float(), target.float()
                # data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                output = cpu_model(data)
                output = torch.squeeze(output)
                pred = torch.round(output).int()
                true_num = pred.eq(target.data.view_as(pred)).sum()
                correct += true_num
                total += len(data)
            
        print("test_accuarcy: ", (correct / total).item(), time_d)

if __name__ == "__main__":
    train()