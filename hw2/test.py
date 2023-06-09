import torch
import time
import torch.nn as nn
import torchtext
import random
torchtext.datasets.imdb.IMDB
train_text = torchtext.data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)
train_label = torchtext.data.LabelField(dtype = torch.float)
train_data, test_data = torchtext.datasets.imdb.IMDB.splits(train_text, train_label)
SEED=20221212
train_data, valid_data = train_data.split(random_state = random.seed(SEED),split_ratio=0.8)
MAX_VOCAB_SIZE = 25000
train_text.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)
train_label.build_vocab(train_data)
BATCH_SIZE = 2128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)
imput_dim = len(train_text.vocab)
vector_dim = 300
hidden_dim = 256
output_dim = 1
layer_num = 2
is_bidirectional = True
dropout = 0.5
pad_idx = train_text.vocab.stoi[train_text.pad_token]
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
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.classifier_head(hidden)
classifier = torch.nn.Sequential(
    nn.Linear(2 * hidden_dim, 1),
    nn.Sigmoid()
)

model = Model(imput_dim, vector_dim, hidden_dim, output_dim, layer_num, is_bidirectional, dropout, pad_idx, classifier)
model.embedding.weight.data.copy_(train_text.vocab.vectors)
unk_idx = train_text.vocab.stoi[train_text.unk_token]
model.embedding.weight.data[unk_idx] = torch.zeros(vector_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(vector_dim)
model.embedding.weight.requires_grad = False
optimizer = torch.optim.Adam(model.parameters())
Loss = torch.nn.BCELoss()
model = model.to(device)
Loss = Loss.to(device)

def cal_acc(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

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
test_loss, test_acc = evaluate(model, test_iterator, Loss)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.5f}')