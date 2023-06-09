#!/usr/bin/env python
# coding: utf-8

# # DeepLearning Assignment 2 实验报告
# # SA22221042 汪泱泱

# ## 一、实验环境

# GPU TITAN Xp  
# CUDA 10.1  
# python 3.7.13  
# torch 1.8.1  
# torchtext 0.6.0  
# spacy 3.4.3

# ## 二、实验过程

# In[1]:


import torch
import time
import torch.nn as nn
import torchtext
import random


# 首先进行数据集的预处理。
# 由于IMDB公开数据集“Large Movie Review Dataset“是非常常见的公开数据集，torchtext中提供了接口`torchtext.datasets.imdb.IMDB`，我们可以直接使用其进行预处理。

# In[2]:


train_text = torchtext.data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)
train_label = torchtext.data.LabelField(dtype = torch.float)


# In[3]:


train_data, test_data = torchtext.datasets.imdb.IMDB.splits(train_text, train_label)


# 划分验证集，划分比例为训练集：验证集=4:1

# In[4]:


SEED=20221212
train_data, valid_data = train_data.split(random_state = random.seed(SEED),split_ratio=0.8)


# 使用facebook预训练好的fasttext.en.300d的词向量编码器构建语料库

# In[5]:


MAX_VOCAB_SIZE = 25000
train_text.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "fasttext.en.300d", 
                 unk_init = torch.Tensor.normal_)
train_label.build_vocab(train_data)


# 按batch_size打包数据

# In[6]:


BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)


# 定义各参数和超参数

# In[7]:


imput_dim = len(train_text.vocab)
vector_dim = 300
hidden_dim = 256
output_dim = 1
layer_num = 3
is_bidirectional = True
dropout = 0.5
pad_idx = train_text.vocab.stoi[train_text.pad_token]


# 定义RNN模型，选择使用双向LSTM网络,使用torch.cat()连接两层隐藏层，并使用dropout防止网络过拟合

# In[18]:


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, classifier):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier_head = classifier
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.classifier_head(hidden)


# 分类头，先简单使用一层全连接层和一层sigmoid将值映射到$[0,1]$上，便于计算二分类交叉熵

# In[19]:


classifier = torch.nn.Sequential(
    nn.Linear(2 * hidden_dim, 1),
    nn.Sigmoid()
)


# 实例化模型，载入预训练的向量

# In[20]:


model = Model(imput_dim, vector_dim, hidden_dim, output_dim, layer_num, is_bidirectional, dropout, pad_idx, classifier)
model.embedding.weight.data.copy_(train_text.vocab.vectors)


# <unk>未知词标记，<pad>填充标记，这两个标记与情感无关，所以填充torch全零向量。

# In[21]:


unk_idx = train_text.vocab.stoi[train_text.unk_token]
model.embedding.weight.data[unk_idx] = torch.zeros(vector_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(vector_dim)
model.embedding.weight.requires_grad = False


# 使用BCELoss作为模型损失函数

# In[22]:


optimizer = torch.optim.Adam(model.parameters())
Loss = torch.nn.BCELoss()
model = model.to(device)
Loss = Loss.to(device)


# 定义计算准确率的函数，对结果取四舍五入近似和真实值比较是否相同

# In[23]:


def cal_acc(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练代码（有反向传播更新参数）和在验证集上的上测试loss和acc的代码：

# In[24]:


def train(model, iterator, optimizer, Loss):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = Loss(predictions, batch.label)
        acc = cal_acc(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, Loss):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = Loss(predictions, batch.label)
            acc = cal_acc(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 开始训练。经过实验，训练可能会在20轮作用收敛，所以我们将训练30个epoch。

# In[ ]:


epochs = 30
best_valid_loss = float('inf')
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_iterator, optimizer, Loss)
    valid_loss, valid_acc = evaluate(model, valid_iterator, Loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.5f}')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc:.5f}')
model.load_state_dict(torch.load('model.pt'))


# ### 三、参数选取

# 下面修改一些网络参数训练网络后，在验证进行测试，以求找到最佳参数。

# 首先改变隐藏层层数

# | Layer Num | Best Valid Loss |
# | ---------- | --------------- |
# | 1         | 0.293          |
# | 2        | **0.263**      |
# | 3        | 0.271          |

# 隐藏层维度

# | Hidden Layer Dimension  | Best Valid Loss |
# | ---------- | --------------- |
# | 128         | 0.285          |
# | 256        | **0.263**      |
# | 512        | 0.266          |

# 词向量嵌入方法

# | Vector Embedding Method | Best Valid Loss |
# | ---------- | --------------- |
# | fasttext.en.300d         | **0.263**          |
# | glove.6B.100d        | 0.274      |

# 批的大小

# | Batch Size | Best Valid Loss |
# | ---------- | --------------- |
# | 64         | 0.281          |
# | 128        | **0.263**      |
# | 256        | 0.274          |

# ### 四、测试结果

# 使用验证集得到的最佳参数，在训练集上训练后，在测试集上进行测试

# In[25]:


imput_dim = len(train_text.vocab)
vector_dim = 300
hidden_dim = 256
output_dim = 1
layer_num = 2
is_bidirectional = True
dropout = 0.5
pad_idx = train_text.vocab.stoi[train_text.pad_token]


# In[26]:


epochs = 30
best_valid_loss = float('inf')
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_iterator, optimizer, Loss)
    valid_loss, valid_acc = evaluate(model, valid_iterator, Loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.5f}')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc:.5f}')


# 选择验证集上表现最好的模型参数在测试集上测试

# In[27]:


test_loss, test_acc = evaluate(model, test_iterator, Loss)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.5f}')


# ACC为0.89143
