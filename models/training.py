import copy
from pytorch_lightning import Trainer
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chord_dataset import CustomChordDataset, DatasetType
from model import Model
from data import CHORD_LABELS

if __name__ == "__main__":
    dataset = CustomChordDataset()

    train = copy.deepcopy(dataset).set_fold(DatasetType.TRAIN)
    test = copy.deepcopy(dataset).set_fold(DatasetType.TEST)
    val = copy.deepcopy(dataset).set_fold(DatasetType.VAL)

    # Start the Trainer
    trainer = Trainer(max_epochs=10,)
    # Define the Model
    model = Model(train, test, val, bsz=32, lr=1e-3, num_classes=len(CHORD_LABELS))
    # Train the Model
    trainer.fit(model)
    # Test on the Test SET, it will print validation
    trainer.test()

    torch.save(model, "models/saved_models/model.pth")
