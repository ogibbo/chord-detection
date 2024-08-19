import copy
from pytorch_lightning import Trainer
import torch

from chord_dataset import CustomChordDataset, DatasetType
from simple_model import SimpleModel


if __name__ == "__main__":
    dataset = CustomChordDataset()

    train = copy.deepcopy(dataset).set_fold(DatasetType.TRAIN)
    test = copy.deepcopy(dataset).set_fold(DatasetType.TEST)
    val = copy.deepcopy(dataset).set_fold(DatasetType.VAL)

    # Start the Trainer
    trainer = Trainer(max_epochs=10,)
    # Define the Model
    model = SimpleModel(train, test, val)
    # Train the Model
    trainer.fit(model)
    # Test on the Test SET, it will print validation
    trainer.test()

    torch.save(model, "models/saved_models/simple_model.pth")
