import torch
import torch.nn as nn
import torch.nn.functional as F
from GCNConv import GCNConv

class GCN_Node_Classification(nn.Module):
    def __init__(self, graph_layer_num, input_size, output_size, layer_size, dropout_rate, activation, not_ppi = True):
        super(GCN_Node_Classification, self).__init__()

        assert graph_layer_num>=2
        self.dropout_rate = dropout_rate
        self.gcn_list = nn.ModuleList()
        self.gcn_list.append(GCNConv(input_size, layer_size))
        self.activation = activation
        for i in range(graph_layer_num-2):
            self.gcn_list.append(GCNConv(layer_size, layer_size))
        self.gcn_list.append(GCNConv(layer_size, output_size))
        self.ppi = not not_ppi

    def forward(self, x, adj):
        for i in range(len(self.gcn_list)):
            x = self.gcn_list[i](x, adj)
            if i != len(self.gcn_list)-1:
                if self.activation == 'relu':
                    x = F.relu(x)
                elif self.activation == 'tanh':
                    x = F.tanh(x)
                else:
                    x = F.sigmoid(x)
                x = F.dropout(x, self.dropout_rate)
        if self.ppi:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

class GCN_Link_Prediction(nn.Module):
    def __init__(self, graph_layer_num, input_size, output_size, layer_size, dropout_rate, activation, not_ppi = True):
        super(GCN_Link_Prediction, self).__init__()

        assert graph_layer_num>=2
        self.dropout_rate = dropout_rate
        self.gcn_list = nn.ModuleList()
        self.gcn_list.append(GCNConv(input_size, layer_size))
        self.activation = activation
        for i in range(graph_layer_num-2):
            self.gcn_list.append(GCNConv(layer_size, layer_size))
        self.gcn_list.append(GCNConv(layer_size, output_size))
        self.ppi = not not_ppi

    def encode(self, x, adj):
        for i in range(len(self.gcn_list)):
            x = self.gcn_list[i](x, adj)
            if i != len(self.gcn_list)-1:
                if self.activation == 'relu':
                    x = F.relu(x)
                elif self.activation == 'tanh':
                    x = F.tanh(x)
                else:
                    x = F.sigmoid(x)
                x = F.dropout(x, self.dropout_rate)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print(dst.size())   # (7284, 64)
        r = (src * dst).sum(dim=-1)
        # print(r.size())   (7284)
        return r

    def forward(self, x, adj, edge_label_index):
        return torch.sigmoid(self.decode(self.encode(x, adj), edge_label_index))
