import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from cmath import pi
from sklearn import datasets
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

num_samples = 10000
train_rate = 0.8
train_samples = int(num_samples * train_rate)

x = np.random.uniform(0, 4*pi, num_samples)
np.random.shuffle(x)
x = x.reshape(-1, 1)
x = torch.tensor(x, dtype=torch.float32)
y = torch.sin(x) + torch.exp(-x)
# print(x)
# print(y)
train_x = x[:train_samples]
train_y = y[:train_samples]
test_x = x[train_samples:]
test_y = y[train_samples:]

batch_size=64
dataset = TensorDataset(train_x, train_y)
data_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1000)
        self.fc3 = nn.Linear(1000, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class XKnet(nn.Module):
    def __init__(self):
        super(XKnet, self).__init__()
        input_size = 1
        hidden_size = 64
        hidden_layers = 4
        num_epochs = 30
        batch_size = 100
        learning_rate = 0.001
        activate = 'relu'
        self.hidden_layers = hidden_layers
        self.activate_fcs = {
            'relu': nn.ReLU(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh()
        }
        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_size, hidden_size))
        for i in range(self.hidden_layers):
            # self.fc_list.append(nn.ReLU())
            self.fc_list.append(self.activate_fcs.get(activate))
            self.fc_list.append(nn.Linear(hidden_size, hidden_size))
        # self.fc_list.append(nn.ReLU())
        self.fc_list.append(self.activate_fcs.get(activate))
        self.fc_list.append(nn.Linear(hidden_size, 1))

    def forward(self, x):
        for fc in self.fc_list:
            x = fc(x)
        return x

model = Model()

n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_ size=20, gamma=0.1)

Loss = nn.MSELoss()

epochs = 100

for epoch in range(epochs):

    epoch_loss = 0

    for batch_id, (x, y) in enumerate(data_loader):
        # print(x.shape)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = Loss(y, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # scheduler.step()
    print('[Epoch {}/{}] Train Loss = {:.4f}, Test Loss = {:.4f}'
    .format(epoch + 1, epochs, Loss(train_y, model(train_x)), Loss(test_y, model(test_x))))

 

x = range(epochs)
plt.plot(x, train_loss, label="train_loss", linewidth=1.5)
plt.plot(x, test_loss, label="test_loss", linewidth=1.5)
plt.plot(x, np.zeros(len(x)), 'red', linestyle='--', linewidth=1)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# print(x)
# print(y)
# torch.sin()