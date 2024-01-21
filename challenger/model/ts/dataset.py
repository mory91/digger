import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def get_sequencet_data(path, target_name='target', split=0.3):
    target = 'target'
    ts = pd.read_csv(path)[100:-100]
    ts = ts[ts['size'] > 0]
    ts = ts.apply(pd.to_numeric)
    ts[target] = ts['size'].shift(-1)
    ts = ts[:-1]
    split_index = int(ts.shape[0] * split)
    df_train = ts[:-split_index]
    df_test = ts[-split_index:]
    return df_train, df_test


def scale_data(df_train, df_test):
    scaler = MinMaxScaler(feature_range=(1, 3)).fit(df_train)
    df_train[df_train.columns] = scaler.transform(df_train)
    df_test[df_test.columns] = scaler.transform(df_test)


class SequenceDataset(Dataset):
    def __init__(self, dataframe, features, target='target', sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class SequenceToSequenceDataset(Dataset):
    def __init__(self, dataframe, features, target='target', sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i):
        x = self.X[i:i + self.sequence_length]
        y = self.y[i + 1:i + self.sequence_length + 1]
        return x, y
