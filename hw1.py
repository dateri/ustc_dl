from turtle import forward
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from cmath import pi
from sklearn import datasets
from torch.utils.data import TensorDataset, DataLoader

num_samples = 10000
train_rate = 0.8

x = np.random.uniform(0, 4*pi, num_samples)
np.random.shuffle(x)
x = torch.tensor(x)
y = torch.sin(x) + torch.exp(-x)

train_x = x[:num_samples*train_rate]
train_y = y[:num_samples*train_rate]
test_x = x[num_samples*train_rate:]
test_y = y[num_samples*train_rate:]

batch_size=64
dataset = TensorDataset(train_x, train_y)
train_iter = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

class Model():
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1000)
        self.fc3 = nn.Linear(1000, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1)
        x = F.relu(self.fc2)
        x = F.relu(self.fc3)
        x = F.relu(self.fc4)
        return forward

print(x)
print(y)
# torch.sin()