import torch
import torch.nn.functional as F
from GCNModel import GCN_Link_Prediction
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

adj, label_num, train_data, val_data, test_data = utils.preprocessing(opt.dataset, 'link_prediction')

graph_layer_num = opt.graph_layer_num
input_size = train_data.x.shape[1]
output_size = label_num
layer_size = opt.layer_size
dropout_rate = 0.5
epochs = 500
lr = opt.lr

sys.stdout = open('./out/'+ os.path.basename(sys.argv[0]).split('.')[0]+ '/' + opt.dataset + '_' + str(graph_layer_num) + '_' + str(layer_size) + '_' + str(opt.activation) + '.log', "w")

model = GCN_Link_Prediction(graph_layer_num=graph_layer_num, input_size=input_size, layer_size=layer_size, output_size=output_size, dropout_rate=dropout_rate, activation=opt.activation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
adj = adj.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

for epoch in range(epochs):
    model.train()
    t = time.time()
    optimizer.zero_grad()
    output = model(train_data.x, adj, train_data.edge_label_index)
    loss_train = F.binary_cross_entropy(output, train_data.edge_label)
    with torch.no_grad():
        auc_train = utils.cal_auc(output, train_data.edge_label)
    loss_train.backward()
    optimizer.step()

    model.eval()
    
    with torch.no_grad():
        output = model(val_data.x, adj, val_data.edge_label_index)
        loss_val = F.binary_cross_entropy(output, val_data.edge_label)
        auc_val = utils.cal_auc(output, val_data.edge_label)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'auc_train: {:.4f}'.format(auc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'auc_val: {:.4f}'.format(auc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    