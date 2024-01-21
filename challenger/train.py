import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    r2_score, mean_absolute_percentage_error
)
from constants import TS_FEATURES, TS2_FEATURES, MINIMAL_FEATURES
from model.ts.dataset import (
    scale_data,
    get_sequencet_data,
    SequenceDataset,
    SequenceToSequenceDataset
)
from model.ts.lstm import ShallowRegressionLSTM
from model.ts.transformer import DecoderOnlyTransfomer

# config
batch_size = 128
sequence_length = 16
lstm_hidden_units = 128
lstm_model_mid_repr = 32
learning_rate = 1e-2
loss_function = nn.MSELoss()
EPOCHS = 20
EPOCHS_FULL = 50
learning_rate_full = 1e-2
FULL_PATH = "files"
main_device = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = {
    'seq': SequenceDataset,
    'seq2seq': SequenceToSequenceDataset
}


def get_datasets(df_train, df_test, features, ds_key='seq2seq'):
    train_dataset = ds[ds_key](
        df_train,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = ds[ds_key](
        df_test,
        features=features,
        sequence_length=sequence_length
    )
    return train_dataset, test_dataset


def get_dataloaders(df_train, df_test, features):
    train_dataset, test_dataset = get_datasets(df_train, df_test, features)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_simple_model(num_features, device, mid_repr=128, hidden_units=256):
    model = ShallowRegressionLSTM(
        num_features=num_features,
        hidden_units=hidden_units,
        n_embd=mid_repr
    ).to(device)
    return model


def get_transformer_model(num_features, device, mid_repr=128):
    model = DecoderOnlyTransfomer(
        block_size=sequence_length,
        num_features=num_features,
        n_embd=mid_repr
    ).to(device)
    return model


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        x, yy = X.cuda(), y.cuda()
        output = model(x).squeeze()
        loss = loss_function(output, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    y_pred = []
    y_real = []

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            x, yy = X.cuda(), y.cuda()
            output = model(x).squeeze()
            total_loss += loss_function(output, yy).item()
            y_pred.extend(output.cpu().numpy())
            y_real.extend(y.numpy())
    avg_loss = total_loss / num_batches
    return (
        avg_loss,
        r2_score(y_real, y_pred),
        mean_absolute_percentage_error(y_real, y_pred)
    )


def train(path_features):
    full_path, features, name = path_features
    times = list(reversed(sorted(map(int, os.listdir(full_path)))))
    result = {}
    trains = []
    tests = []
    global learning_rate
    if name == "full":
        num_epochs = EPOCHS_FULL
        learning_rate = learning_rate_full
    else:
        num_epochs = EPOCHS
    steps = list(range(num_epochs))
    for td in times:
        path = f"{full_path}/{td}/full.csv"
        df_train, df_test = get_sequencet_data(path, features, split=0.3)
        scale_data(df_train, df_test)
        train_loader, test_loader = get_dataloaders(
            df_train,
            df_test,
            features
        )
        model = get_transformer_model(
            len(features),
            device=main_device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        main_test_loss = float('inf')
        main_r2, main_mape, main_train_loss = -1, -1, -1
        print("STARTED", td)
        if td < 5000:
            num_epochs = 10  # CHECK
        for ix_epoch in range(num_epochs):
            train_loss = train_model(
                train_loader,
                model,
                loss_function,
                optimizer=optimizer
            )
            test_loss, r2, mape = test_model(
                test_loader,
                model,
                loss_function
            )
            print(test_loss, mape)
            tests.append(test_loss)
            trains.append(train_loss)
            if main_test_loss > test_loss:
                main_test_loss = test_loss
                main_train_loss = train_loss
                main_mape = mape
                main_r2 = r2
        result[td] = {
            'test_mse': float(main_test_loss),
            'r2': float(main_r2),
            'train_mse': float(main_train_loss),
            'mape': float(main_mape)
        }
        print(td, result[td])
    return result, trains, tests, steps


if __name__ == "__main__":
    paths_features = [
        (FULL_PATH, MINIMAL_FEATURES, "minimal"),
        (FULL_PATH, TS_FEATURES, "full"),
        (FULL_PATH, TS2_FEATURES, "limited")
    ]
    result = {}
    trains_log = {}
    for pfn in paths_features:
        _, _, name = pfn
        r, tr, te, st = train(pfn)
        result[name] = r
        trains_log[name] = {"tr": tr, "te": te, "st": st}
    result_file = open('ts_result.json', 'w')
    json.dump(result, result_file)
    result_file = open('tr.json', 'w')
    json.dump(trains_log, result_file)
