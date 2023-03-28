import torch
import pytorch_lightning as pl


class SimpleModel(pl.LightningModule):
    def __init__(self, criterion, lr: float = 0.01, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.get_step_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_step_loss(batch, batch_idx)
        return loss

    def validation_epoch_end(self, outs):
        loss = torch.stack(outs).mean()
        self.log("val_loss", loss)
        print("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)


class SimpleForwardModel(SimpleModel):
    def get_step_loss(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        return loss


class TeacherForcedModel(SimpleModel):
    def get_step_loss(self, batch, batch_idx):
        x, y = batch
        out = self(x, y)
        loss = self.criterion(out, y)
        return loss
