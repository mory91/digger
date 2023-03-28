import torch
import torch.nn.functional as F
from torch import nn
from model.abstract import SimpleForwardModel


def smape(real, prediction):
    epsilon = 0.1
    true_o = real
    pred_o = prediction
    summ = torch.maximum(
        torch.abs(true_o) + torch.abs(pred_o) + epsilon,
        torch.ones_like(true_o) * (0.5 + epsilon))
    smape = torch.abs(pred_o - true_o) / summ * 2.0
    return torch.sum(smape)


def mae(real, prediction):
    return nn.functional.l1_loss(real, prediction)


def mse(real, prediction):
    return nn.functional.mse_loss(real, prediction)


def eval_model(real, prediction):
    return {
        'mse': mse(real, prediction).item(),
        'mae': mae(real, prediction).item(),
        'smape': smape(real, prediction).item()
    }


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


class SimpleLSTM(SimpleForwardModel):
    def __init__(
            self,
            num_features,
            hidden_units,
            out_features=1,
            criterion=F.mse_loss,
            **kwargs
    ):
        super().__init__(criterion=criterion, **kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=out_features)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        out = self.linear(out)
        return out


class MultiLayerLSTM(SimpleForwardModel):
    def __init__(self, num_features, sizes, out_features, num_layers=1):
        super().__init__()
        self.container = nn.ModuleList()
        self.my_sizes = [num_features] + sizes
        self.num_features = num_features
        self.num_layers = num_layers
        for idx, s in enumerate(self.my_sizes[1:]):
            self.container.append(
                nn.GRU(
                    input_size=self.my_sizes[idx],
                    hidden_size=s,
                    batch_first=True,
                    num_layers=self.num_layers
                )
            )
        self.linear = nn.Linear(sizes[-1], out_features)

    def forward(self, x):
        for idx, layer in enumerate(self.container):
            h_t = torch.zeros(
                self.num_layers, x.size(0), self.my_sizes[idx + 1]
            ).to(x)
            x, h = layer(x, h_t)
        out = self.linear(h[-1])
        return out
