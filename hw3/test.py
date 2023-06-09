import random
import sys
import time
import torch
import torch.nn as nn
import torchtext
import tqdm
from transformers import AutoTokenizer, AutoModel


pretrained_model_name = 'bert-large-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
bertModel = AutoModel.from_pretrained(pretrained_model_name)
MAX_TOKENS = tokenizer.max_model_input_sizes[pretrained_model_name]-2
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence, max_length=MAX_TOKENS, truncation=True)
    return tokens
train_text = torchtext.data.Field(batch_first=True,
                            use_vocab=False,
                            tokenize = tokenize_and_cut,
                            preprocessing = tokenizer.convert_tokens_to_ids,
                            init_token=tokenizer.cls_token_id,
                            eos_token=tokenizer.sep_token_id,
                            pad_token=tokenizer.pad_token_id,
                            unk_token=tokenizer.unk_token_id)
train_label = torchtext.data.LabelField(dtype = torch.float)
train_data, test_data = torchtext.datasets.imdb.IMDB.splits(train_text, train_label)
SEED=20230102
train_data, valid_data = train_data.split(random_state = random.seed(SEED),split_ratio=0.25)
train_label.build_vocab(train_data)
class SentimentAnalysisModel(nn.Module):
    def __init__(self, bertModel):
        super().__init__()
        self.bertModel = bertModel
        self.classifier = torch.nn.Sequential(nn.Linear(bertModel.config.hidden_size, 1), nn.Sigmoid())
        
    def forward(self, text):
        embedded = self.bertModel(text)[0]
        predict = self.classifier(embedded.mean(dim=1))
        return predict
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)
model = SentimentAnalysisModel(bertModel)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)
Loss = torch.nn.BCELoss()
model = nn.DataParallel(model).to(device)
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
    for batch in tqdm.tqdm(iterator, desc='training...', file=sys.stdout):
        optimizer.zero_grad()
        text = batch.text
        predictions = model(text).squeeze(1)
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
        for batch in tqdm.tqdm(iterator, desc='evaluating...', file=sys.stdout):
            text = batch.text
            predictions = model(text).squeeze(1)
            loss = Loss(predictions, batch.label)
            acc = cal_acc(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
epochs = 2
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