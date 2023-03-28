import torch
from torch import nn
import torch.nn.functional as F
import random

from model.abstract import SimpleForwardModel, TeacherForcedModel


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(
            input_dim, hid_dim, n_layers, batch_first=True
        )

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.rnn = nn.LSTM(
            output_dim, hid_dim, n_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(-1)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(TeacherForcedModel):
    def __init__(
            self,
            encoder,
            decoder,
            criterion=F.mse_loss
    ):
        super().__init__(criterion)
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(
            batch_size, trg_len, trg_vocab_size
        ).to(src)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        teacher_forcing_ratio = teacher_forcing_ratio if self.training else 0.0

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output
            input = trg[:, t] if teacher_force else top1
        return outputs
