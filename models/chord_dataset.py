import torch
from torch.utils.data import Dataset
import pickle
from random import shuffle
from enum import Enum

# Make simple Enum for code clarity
class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


class CustomChordDataset(Dataset):
    def __init__(self, file_path="data/processed_data/all_data.pkl"):

        with open(file_path, "rb") as f:
            self.data = pickle.load(f)

        shuffle(self.data)

        train_split = 0.5
        val_split = 0.2

        train_end = int(
            train_split * len(self.data)
        )
        val_end = train_end + int(
            val_split * len(self.data)
        )

        self.train_data = self.data[:train_end]
        self.val_data = self.data[train_end:val_end]
        self.test_data = self.data[val_end:]

        self.train = torch.stack([tensor for tensor, _ in self.train_data])
        self.val = torch.stack([tensor for tensor, _ in self.val_data])
        self.test = torch.stack([tensor for tensor, _ in self.test_data])

        self.train_labels = torch.tensor([label for _, label in self.train_data])
        self.val_labels = torch.tensor([label for _, label in self.val_data])
        self.test_labels = torch.tensor([label for _, label in self.test_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def set_fold(self, set_type):
        if set_type == DatasetType.TRAIN:
            self.dataset, self.labels = self.train, self.train_labels
        if set_type == DatasetType.TEST:
            self.dataset, self.labels = self.test, self.test_labels
        if set_type == DatasetType.VAL:
            self.dataset, self.labels = self.val, self.val_labels

        return self
