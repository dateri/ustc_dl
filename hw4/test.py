import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch._C import parse_ir
import torch_geometric
from torch_geometric import datasets
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

path = "./dataset/citeseer/"
cites = path + "citeseer.cites"
content = path + "citeseer.content"

# 索引字典，将原本的论文id转换到从0开始编码
index_dict = dict()
# 标签字典，将字符串标签转化为数值
label_to_index = dict()

features = []
labels = []
edge_index = []

with open(content,"r") as f:
    nodes = f.readlines()
    for node in nodes:
        node_info = node.split()
        index_dict[int(node_info[0])] = len(index_dict)
        features.append([int(i) for i in node_info[1:-1]])
        
        label_str = node_info[-1]
        if(label_str not in label_to_index.keys()):
            label_to_index[label_str] = len(label_to_index)
        labels.append(label_to_index[label_str])

with open(cites,"r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.split()
        # 训练时将边视为无向的，但原本的边是有向的，因此需要正反添加两次
        edge_index.append([index_dict[int(start)],index_dict[int(end)]])
        edge_index.append([index_dict[int(end)],index_dict[int(start)]])


class GCNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label):
        super(GCNNet,self).__init__()
        self.GCN1 = GCNConv(num_feature, 16)
        self.GCN2 = GCNConv(16, num_label)  
        self.dropout = torch.nn.Dropout(p=0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        
        return F.log_softmax(x, dim=1)



# 为每个节点增加自环，但后续GCN层默认会添加自环，跳过即可
# for i in range(2708):
#     edge_index.append([i,i])
  
# 转换为Tensor
labels = torch.LongTensor(labels)
features = torch.FloatTensor(features)
# 行归一化
features = torch.nn.functional.normalize(features, p=1, dim=1)
edge_index =  torch.LongTensor(edge_index)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 本电脑只有一个GPU

mask = torch.randperm(len(index_dict)) # 随机打乱顺序
train_mask = mask[:140]
val_mask = mask[140:640]
test_mask = mask[1708:2708]

for i in features[0]:
    if i != 0:
        print(i)
print(labels)
citeseer = Data(x = features, edge_index = edge_index.t().contiguous(), y = labels).to(device)

def gcn_apply():
    model = GCNNet(features.shape[1], len(label_to_index)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    acc_record = []
    loss_record = []

    for epoch in range(500):
        optimizer.zero_grad()
        out = model(citeseer)
        loss = F.nll_loss(out[train_mask], citeseer.y[train_mask])
        loss_record.append(loss.item())
        print('epoch: %d loss: %.4f' %(epoch, loss))
        loss.backward()
        optimizer.step()
        
        # if((epoch + 1)% 10 == 0):
        model.eval()
        _, pred = model(citeseer).max(dim=1)
        correct = int(pred[test_mask].eq(citeseer.y[test_mask]).sum().item())
        acc = correct / len(test_mask)
        acc_record.append(acc)
        print('Accuracy: {:.4f}'.format(acc))
        model.train()
            




if __name__ == "__main__":
    gcn_apply()
