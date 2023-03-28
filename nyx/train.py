import os
import json
import sys

import torch
from torch import utils
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from data.dataset import SineDataset
from data.clean import get_datatrace, get_rawdata
from model.models import eval_model
from model.seq2seq import (
    Encoder,
    Decoder,
    Seq2Seq,
)
from model.models import (
    SimpleLSTM
)

BATCH_SIZE = 1024
EPOCHS = 500

CELLAR = "data/cellar"


def train(model, train_ds, test_ds, batch_size=BATCH_SIZE, epochs=EPOCHS):
    train_loader = utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False
    )
    val_loader = utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(), version=2, name="lightning_logs"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        check_val_every_n_epoch=10,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    trainer.validate(
        model=model,
        ckpt_path='best',
        dataloaders=val_loader
    )
    predictions = trainer.predict(
        model=model,
        ckpt_path='best',
        dataloaders=val_loader
    )

    test_loader = utils.data.DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
    )
    x, real = [batch for batch in test_loader][0]
    predictions = torch.cat(predictions, dim=0)
    return eval_model(real=real, prediction=predictions)


def try_cros():
    sequence_length = 128
    evals = []
    train_ds, test_ds = get_rawdata(
        sequence_length, filename=f"{CELLAR}/croston.csv", split=0.5
    )
    model = SimpleLSTM(2, 64, 2, lr=0.1)
    eval = train(model, train_ds, test_ds, len(train_ds), epochs=2000)
    evals.append(eval)
    json.dump({'evals': evals}, open('result.json', 'w'))


def try_seqr():
    sequence_length = 128
    evals = []
    train_ds, test_ds = get_datatrace(1000, sequence_length, split=0.4)
    model = SimpleLSTM(2, 16, 1, lr=0.01)
    eval = train(model, train_ds, test_ds, 128, epochs=100)
    evals.append(eval)
    json.dump({'evals': evals}, open('result.json', 'w'))


def try_seqs():
    sequence_length = 256
    times = list(range(200, 3000, 250))
    evals = []
    for time in times:
        train_ds, test_ds = get_datatrace(time, sequence_length)
        train_loader = utils.data.DataLoader(
            train_ds,
            batch_size=1024,
            num_workers=16,
            shuffle=False
        )
        val_loader = utils.data.DataLoader(
            test_ds,
            batch_size=1024,
            num_workers=16,
            shuffle=False,
        )
        logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(), version=2, name="lightning_logs"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1
        )
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=1,
            logger=logger,
            check_val_every_n_epoch=10,
            callbacks=[checkpoint_callback]
        )
        trainer.fit(
            model=simple_lstm,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        trainer.validate(
            model=simple_lstm,
            ckpt_path='best',
            dataloaders=val_loader
        )
        predictions = trainer.predict(
            model=simple_lstm,
            ckpt_path='best',
            dataloaders=val_loader
        )

        test_loader = utils.data.DataLoader(
            test_ds,
            batch_size=len(test_ds),
            shuffle=False,
        )
        x, real = [batch for batch in test_loader][0]
        predictions = torch.cat(predictions, dim=0)
        evals.append(eval_model(real=real, prediction=predictions))
    json.dump({'times': times, 'evals': evals}, open('result.json', 'w'))


def try_sin():
    train_ds = SineDataset(torch.arange(0, 100000, 0.5))
    test_ds = SineDataset(torch.arange(100000, 102000, 0.5))
    INPUT_DIM = 2
    OUTPUT_DIM = 1
    HID_DIM = 32
    N_LAYERS = 1
    enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS)
    dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS)
    model = Seq2Seq(enc, dec)
    train(model, train_ds, test_ds, 2048, 3000)


commands = {
    "sin": try_sin,
    "seqs": try_seqs,
    "seqr": try_seqr,
    "cros": try_cros
}

if __name__ == "__main__":
    t = sys.argv[1]
    commands[t]()
