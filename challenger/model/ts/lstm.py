from torch import nn
import torch.nn.functional as F
import torch


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, n_embd, hidden_units):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.fc_1 = nn.Linear(self.num_features, n_embd)
        self.fc_2 = nn.Linear(self.hidden_units, n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)
        self.linear = nn.Linear(in_features=n_embd, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.ln(F.relu(self.fc_1(x)))
        x = self.dropout_1(x)

        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_().cuda()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_().cuda()

        out, (hn, _) = self.lstm(x, (h0, c0))

        out = F.relu(self.fc_2(self.dropout_2(out)))
        out = self.linear(out)

        return out


class SimpleLSTM(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.linear = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_().cuda()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_().cuda()

        out, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(out)

        return out
