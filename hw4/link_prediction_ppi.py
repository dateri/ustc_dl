import torch
import torch.nn.functional as F
from GCNModel import GCN_Link_Prediction
import torch.optim as optim
import time
import utils
import argparse
import sys
import os

adj_list, train_data_list, val_data_list, test_data_list = utils.preprocessing('PPI', 'link_prediction')

parser = argparse.ArgumentParser()
parser.add_argument("-graph_layer_num", default=2, type=int)
parser.add_argument("-layer_size", default=256, type=int)
parser.add_argument("-lr", default=1e-3, type=float)
parser.add_argument("-activation", default='relu', type=str)
opt = parser.parse_args()

graph_layer_num = opt.graph_layer_num
input_size = train_data_list[0].x.shape[1]
output_size = 50
layer_size = opt.layer_size
dropout_rate = 0.5
epochs = 500
lr = opt.lr

sys.stdout = open('./out/'+ os.path.basename(sys.argv[0]).split('.')[0] + '/' + str(graph_layer_num) + '_' + str(layer_size) + '_' + str(opt.activation) + '.log', "w")

model = [0] * len(train_data_list)
optimizer = [0] * len(train_data_list)
for i in range(len(train_data_list)):
    model[i] = GCN_Link_Prediction(graph_layer_num=graph_layer_num, input_size=input_size, layer_size=layer_size, output_size=output_size, dropout_rate=dropout_rate, not_ppi=False, activation=opt.activation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model[i].to(device)
    optimizer[i] = optim.Adam(model[i].parameters(), lr=lr, weight_decay=5e-4)    
# def data_cuda(dataset):
#     for i in range(len(dataset)):
#         adj = dataset[i][0].to(device)
#         features = dataset[i][1].to(device)
#         labels = dataset[i][2].to(device)
#         dataset[i] = (adj, features, labels)
# data_cuda(train_dataset)
# data_cuda(val_dataset)
# data_cuda(test_dataset)


for epoch in range(epochs):
    t = time.time()
    total_train_loss, total_train_auc = 0.0, 0.0
    total_train_samples = 0

    for i in range(len(train_data_list)):  
        model[i].train()
        optimizer[i].zero_grad()
        output = model[i](train_data_list[i].x.to(device), adj_list[i].to(device), train_data_list[i].edge_label_index.to(device))
        loss_train = F.binary_cross_entropy(output, train_data_list[i].edge_label.to(device))
        with torch.no_grad():
            auc_train = utils.cal_auc(output, train_data_list[i].edge_label)
        total_train_loss += loss_train * train_data_list[i].x.shape[0]
        total_train_auc += auc_train * train_data_list[i].x.shape[0]
        total_train_samples += train_data_list[i].x.shape[0]
        loss_train.backward()
        optimizer[i].step()

        
    total_val_loss, total_val_auc = 0.0, 0.0
    total_val_samples = 0
    for i in range(len(val_data_list)):
        model[i].eval()
        output = model[i](val_data_list[i].x.to(device), adj_list[i].to(device), val_data_list[i].edge_label_index.to(device))
        loss_val = F.binary_cross_entropy(output, val_data_list[i].edge_label.to(device))
        with torch.no_grad():
            auc_val = utils.cal_auc(output, val_data_list[i].edge_label)
        total_val_loss += loss_val * val_data_list[i].x.shape[0]
        total_val_auc += auc_val * val_data_list[i].x.shape[0]
        total_val_samples += val_data_list[i].x.shape[0]

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(total_train_loss / total_train_samples),
          'auc_train: {:.4f}'.format(total_train_auc / total_train_samples),
          'loss_val: {:.4f}'.format(total_val_loss / total_val_samples),
          'auc_val: {:.4f}'.format(total_val_auc / total_val_samples),
          'time: {:.4f}s'.format(time.time() - t))
