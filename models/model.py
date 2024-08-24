import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


class Model(LightningModule):
    def __init__(self, train, test, val, bsz, lr, num_classes):

        super().__init__()

        self.train_ds = train
        self.val_ds = val
        self.test_ds = test

        self.bsz = bsz
        self.lr = lr

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        z = self.model(x)
        return F.log_softmax(z, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, print_str="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bsz, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bsz, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bsz, shuffle=False)
