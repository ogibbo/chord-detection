import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

BATCH_SIZE = 32

class SimpleModel(LightningModule):
    def __init__(self, train, test, val):
        
        super().__init__()
        
        self.train_ds=train
        self.val_ds=val
        self.test_ds=test

        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4),
        )

        self.accuracy = Accuracy(task='multiclass', num_classes=4)

    def forward(self, x):
        z = self.model(x)
        return F.log_softmax(z, dim=1)
    
    # Same as above
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        return loss
    
    # Make use of the validation set
    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx,print_str='test')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    #
    # HERE: We define the 3 Dataloaders, only train needs to be shuffled
    # This will then directly be usable with Pytorch Lightning to make a super quick model
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE,shuffle=False)