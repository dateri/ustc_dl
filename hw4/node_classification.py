import torch
import torch.nn.functional as F
from GCNModel import GCN_Node_Classification
import torch.optim as optim
import time
import utils
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("-graph_layer_num", default=2, type=int)
parser.add_argument("-layer_size", default=256, type=int)
parser.add_argument("-lr", default=1e-3, type=float)
parser.add_argument("-activation", default='relu', type=str)
parser.add_argument("-dataset", default='cora', type=str)
opt = parser.parse_args()

adj, features, labels, label_num, train_dataset, val_dataset, test_dataset = utils.preprocessing(opt.dataset)

graph_layer_num = opt.graph_layer_num
input_size = features.shape[1]
output_size = label_num
layer_size = opt.layer_size
dropout_rate = 0.5
epochs = 2000
lr = opt.lr

sys.stdout = open('./out/'+ os.path.basename(sys.argv[0]).split('.')[0] + '/' + opt.dataset + '_' + str(graph_layer_num) + '_' + str(layer_size) + '_' + str(opt.activation) + '.log', "w")

model = GCN_Node_Classification(graph_layer_num=graph_layer_num ,input_size=input_size, layer_size=layer_size, output_size=output_size, dropout_rate=dropout_rate, activation=opt.activation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
train_dataset = train_dataset.to(device)
val_dataset = val_dataset.to(device)
test_dataset = test_dataset.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

for epoch in range(epochs):
    model.train()
    t = time.time()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[train_dataset], labels[train_dataset])
    acc_train = utils.cal_acc(output[train_dataset], labels[train_dataset])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[val_dataset], labels[val_dataset])
    acc_val = utils.cal_acc(output[val_dataset], labels[val_dataset])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
