import torch
from torch import nn
import numpy as np
import time
from matplotlib import pyplot as plt
import torch.utils.data as Data

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# set random seed
def setup_seed(seed):
     torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True


# In[2]:


# Data preparation
num_inputs = 1
num_examples = 10000
rate_test = 0.3
print(np.random.rand(num_examples, num_inputs))
x_features = torch.tensor(np.random.rand(num_examples, num_inputs)*4*torch.pi, dtype=torch.float)
y_labels = torch.sin(x_features)
# y_labels += torch.tensor(np.random.normal(0, 0.01, size=y_labels.size()), dtype=torch.float)
# Train_set
trainfeatures = x_features[round(num_examples*rate_test):]
trainlabels = y_labels[round(num_examples*rate_test):]
print(trainfeatures.shape)
# Test_set
testfeatures = x_features[:round(num_examples*rate_test)]
testlabels = y_labels[:round(num_examples*rate_test)]
print(testfeatures.shape)


# In[3]:


# 读取数据
batch_size = 100
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(trainfeatures, trainlabels)
# 把 dataset 放入 DataLoader
train_iter = Data.DataLoader(
    dataset=dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers=0,  # 多线程来读数据， 注意在Windows下需要设置为0
)
# 将测试数据的特征和标签组合
dataset = Data.TensorDataset(testfeatures, testlabels)
# 把 dataset 放入 DataLoader
test_iter = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,  
    num_workers=0,  
)

# In[4]:


# Fully connected neural network
class XKnet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, activate):
        super(XKnet, self).__init__()
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


# In[5]:


setup_seed(20211030)

# Hyper-parameters
input_size = 1
hidden_size = 64
hidden_layers = 4
num_epochs = 30
batch_size = 100
learning_rate = 0.001
activate = 'relu'

# Instantiation the model
xknet = XKnet(input_size, hidden_size, hidden_layers, activate)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(xknet.parameters(), lr=learning_rate)


# In[6]:


# Train the model
t = time.time()
train_loss, test_loss = [], []
for epoch in range(num_epochs):
    for X, y in train_iter:  # x和y分别是小批量样本的特征和标签
        optimizer.zero_grad()
        y_hat = xknet(X)
        loss = criterion(y, y_hat)
        loss.backward()
        optimizer.step()
    train_loss.append(criterion(xknet(trainfeatures), trainlabels).item())
    test_loss.append(criterion(xknet(testfeatures), testlabels).item())
    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], train_loss: {:.6f}, test_loss: {:.6f}'
              .format(epoch+1, num_epochs, train_loss[epoch], test_loss[epoch]))
print('run_time: ', time.time()-t, 's')

x = range(num_epochs)
plt.plot(x, train_loss, label="train_loss", linewidth=1.5)
plt.plot(x, test_loss, label="test_loss", linewidth=1.5)
plt.plot(x, np.zeros(len(x)), 'red', linestyle='--', linewidth=1)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()