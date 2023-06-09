import torch
import torch.nn.functional as F
import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.datasets import PPI

def read_data(dataset_name):
    assert(dataset_name in ['cora', 'citeseer'])
    cites_path = os.path.join('./dataset', dataset_name, dataset_name+'.cites')
    content_path = os.path.join('./dataset', dataset_name, dataset_name+'.content')
    labels = []
    features = []
    ids = []
    paper_reindex = dict()
    label_reindex = dict()
                
    with open(content_path ,"r") as f:
        node_lines = f.readlines()
        index_cnt = 0
        label_cnt = 0
        for line in node_lines:
            node = line.strip('\n').split('\t')
            if node[0] not in paper_reindex:
                paper_reindex[node[0]] = index_cnt
                index_cnt += 1
            if node[-1] not in label_reindex:
                label_reindex[node[-1]] = label_cnt
                label_cnt += 1
        for line in node_lines:
            node = line.strip('\n').split('\t')
            ids.append(paper_reindex[node[0]])
            features.append(node[1: -1])
            labels.append(node[-1])
        x = np.zeros(shape=(index_cnt, len(features[0])), dtype=int)
        y = np.zeros(shape=(index_cnt, 1), dtype=int)
        for i in range(len(ids)):
            for j in range(len(features[i])):
                x[i][j] = int(features[i][j])
            y[i][0] = label_reindex[labels[i]]
            
    with open(cites_path,"r") as f:
        edge_lines = f.readlines()
        edge_index = np.zeros(shape=(2, len(edge_lines)*2), dtype=int)
        edge_num = 0
        for line in edge_lines:
            edge = line.strip('\n').split('\t')
            if edge[0] not in paper_reindex or edge[1] not in paper_reindex:
                continue
            edge_index[0][edge_num] = paper_reindex[edge[0]]
            edge_index[1][edge_num] = paper_reindex[edge[1]]
            edge_num += 1
            edge_index[0][edge_num] = paper_reindex[edge[1]]
            edge_index[1][edge_num] = paper_reindex[edge[0]]
            edge_num += 1
        edge_index = edge_index[:, :edge_num]
    return x, edge_index, y, len(label_reindex)

def get_adj(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
    return adj

def cal_acc(y_pred, y_true):
    return (y_pred.max(1)[1] == y_true).type(torch.float32).mean()

def cal_f1(y_pred, y_true):
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    pred = np.zeros_like(y_pred)
    pred[y_pred > 0.5] = 1
    return f1_score(y_pred=pred, y_true=y_true, average='micro')

def cal_auc(y_pred, y_true):
    return roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
def feature_normalize(x):
    return F.normalize(x, p=1, dim=1)

# D^(-1/2)*A*D^(-1/2)
def adj_normalize(adj):
    row_sum = np.array(adj.sum(axis=1))
    D_inv_sqrt = np.power(row_sum, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = sp.diags(D_inv_sqrt)
    adj_normalized = D_mat_inv_sqrt * adj * D_mat_inv_sqrt
    return adj_normalized

def preprocessing(dataset_name, task='node_classification'):
    if dataset_name == 'PPI':
        return preprocessing_ppi(task)
    features, edge_index, labels, label_num = read_data(dataset_name)
    adj =  get_adj(edges=edge_index, labels=labels)
    adj = adj_normalize(adj + sp.eye(m=adj.shape[0], n=adj.shape[0]))
    N_nodes = features.shape[0]
    idx_train = range(300)
    idx_val = range(300, 800)
    idx_test = range(int(N_nodes*0.9), N_nodes)

    features = torch.FloatTensor(features)
    features = feature_normalize(features)
    labels = torch.LongTensor(labels)
    labels = labels.squeeze()
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if task == 'link_prediction':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = Data(x=features.to(device), edge_index=torch.LongTensor(edge_index).to(device), y=labels.squeeze().to(device))
        transform = RandomLinkSplit(num_val=0.1, num_test=0.1)
        train_data, val_data, test_data = transform(data)
        return adj, label_num, train_data, val_data, test_data

    train_dataset = torch.LongTensor(idx_train)
    val_dataset = torch.LongTensor(idx_val)
    test_dataset = torch.LongTensor(idx_test)

    return adj, features, labels, label_num, train_dataset, val_dataset, test_dataset

def preprocessing_ppi(task='node_classification'):
    path = './ppi/'
    if task == 'link_prediction':
        adj_list, train_data_list, val_data_list, test_data_list = [], [], [], []
        for data in PPI(path):
            transform = RandomLinkSplit(num_val=0.1, num_test=0.1)
            train_data, val_data, test_data = transform(data)
            adj = get_adj(edges=data.edge_index, labels=data.y)
            adj = adj_normalize(adj + sp.eye(m=adj.shape[0], n=adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj_list.append(adj)
            train_data_list.append(train_data)
            val_data_list.append(val_data)
            test_data_list.append(test_data)
        return adj_list, train_data_list, val_data_list, test_data_list
    train_ppi = PPI(path, split = 'train')
    val_ppi = PPI(path, split = 'val')
    test_ppi = PPI(path, split = 'test')
    train_dataset, val_dataset, test_dataset = [], [], []
    for data in train_ppi:
        train_adj = get_adj(edges=data.edge_index, labels=data.y)
        train_adj = adj_normalize(train_adj + sp.eye(m=train_adj.shape[0], n=train_adj.shape[0]))
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj)
        train_features = feature_normalize(data.x)
        train_labels = data.y
        train_dataset.append((train_adj, train_features, train_labels))
    for data in val_ppi:
        val_adj = get_adj(edges=data.edge_index, labels=data.y)
        val_adj = adj_normalize(val_adj + sp.eye(m=val_adj.shape[0], n=val_adj.shape[0]))
        val_adj = sparse_mx_to_torch_sparse_tensor(val_adj)
        val_features = feature_normalize(data.x)
        val_labels = data.y
        val_dataset.append((val_adj, val_features, val_labels))
    for data in test_ppi:
        test_adj = get_adj(edges=data.edge_index, labels=data.y)
        test_adj = adj_normalize(test_adj + sp.eye(m=test_adj.shape[0], n=test_adj.shape[0]))
        test_adj = sparse_mx_to_torch_sparse_tensor(test_adj)
        test_features = feature_normalize(data.x)
        test_labels = data.y
        test_dataset.append((test_adj, test_features, test_labels))
    return train_dataset, val_dataset, test_dataset

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def negative_sample(train_data):
#     # 从训练集中采样与正边相同数量的负边
#     neg_edge_index = negative_sampling(
#         edge_index=train_data.edge_index,
#         num_nodes=train_data.num_nodes,
#         num_neg_samples=train_data.edge_label_index.size(1),
#         method='sparse')
#     # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
#     edge_label_index = torch.cat(
#         [train_data.edge_label_index, neg_edge_index],
#         dim=-1,
#     )
#     edge_label = torch.cat([
#         train_data.edge_label,
#         train_data.edge_label.new_zeros(neg_edge_index.size(1))
#     ], dim=0)

#     return edge_label, edge_label_index