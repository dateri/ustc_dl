import torch
import torch.nn.functional as F
from GCNModel import GCN_Node_Classification
import torch.optim as optim
import time
import utils
import argparse
import sys
import os

train_dataset, val_dataset, test_dataset = utils.preprocessing_ppi()

parser = argparse.ArgumentParser()
parser.add_argument("-graph_layer_num", default=2, type=int)
parser.add_argument("-layer_size", default=256, type=int)
parser.add_argument("-lr", default=1e-3, type=float)
parser.add_argument("-activation", default='relu', type=str)
opt = parser.parse_args()

graph_layer_num = opt.graph_layer_num
input_size = train_dataset[0][1].shape[1]
output_size = train_dataset[0][2].shape[1]
layer_size = opt.layer_size
dropout_rate = 0.5
epochs = 500
lr = opt.lr

sys.stdout = open('./out/'+ os.path.basename(sys.argv[0]).split('.')[0] + '/' + str(graph_layer_num) + '_' + str(layer_size) + '_' + str(opt.activation) + '.log', "w")

model = GCN_Node_Classification(graph_layer_num=graph_layer_num ,input_size=input_size, layer_size=layer_size, output_size=output_size, dropout_rate=dropout_rate, not_ppi=False, activation=opt.activation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def data_cuda(dataset):
    for i in range(len(dataset)):
        adj = dataset[i][0].to(device)
        features = dataset[i][1].to(device)
        labels = dataset[i][2].to(device)
        dataset[i] = (adj, features, labels)
data_cuda(train_dataset)
data_cuda(val_dataset)
data_cuda(test_dataset)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

for epoch in range(epochs):
    model.train()
    t = time.time()
    total_train_loss, total_train_f1 = 0.0, 0.0
    total_train_samples = 0
    for i in range(len(train_dataset)):
        optimizer.zero_grad()
        output = model(train_dataset[i][1], train_dataset[i][0])
        loss_train = F.binary_cross_entropy(output, train_dataset[i][2])
        f1_train = utils.cal_f1(output, train_dataset[i][2])
        total_train_loss += loss_train * train_dataset[i][1].shape[0]
        total_train_f1 += f1_train * train_dataset[i][1].shape[0]
        total_train_samples += train_dataset[i][1].shape[0]
        # print(loss_train, acc_train, train_dataset[i][1].shape[0])
        loss_train.backward()
        optimizer.step()

    total_val_loss, total_val_f1 = 0.0, 0.0
    total_val_samples = 0
    model.eval()
    for i in range(len(val_dataset)):
        output = model(val_dataset[i][1], val_dataset[i][0])
        loss_val = F.binary_cross_entropy(output, val_dataset[i][2])
        acc_val = utils.cal_f1(output, val_dataset[i][2])
        total_val_loss += loss_val * val_dataset[i][1].shape[0]
        total_val_f1 += acc_val * val_dataset[i][1].shape[0]
        total_val_samples += val_dataset[i][1].shape[0]

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(total_train_loss / total_train_samples),
          'f1_train: {:.4f}'.format(total_train_f1 / total_train_samples),
          'loss_val: {:.4f}'.format(total_val_loss / total_val_samples),
          'f1_val: {:.4f}'.format(total_val_f1 / total_val_samples),
          'time: {:.4f}s'.format(time.time() - t))
