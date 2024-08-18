import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(100, 50)  # Example layer
        self.fc2 = nn.Linear(50, 10)   # Example layer
        self.fc3 = nn.Linear(10, 4)    # Example output layer for 4 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x