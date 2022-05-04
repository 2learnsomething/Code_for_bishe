import torch.nn.functional as F
import torch.optim as optim
import torch
import os
import sys

sys.path.append('.')
from language_train.Dataset import test_model, train_model, dataset
from language_model.LSTM import LSTM
from utils import path

n_gpus = 4

train_iter, test_iter, NEWS = dataset('both')
model = LSTM(len(NEWS.vocab), hidden_size=128, num_layers=2, classes=2)

weight_matrix = NEWS.vocab.vectors
model.embedding.weight.data.copy_(weight_matrix)
model = torch.nn.DataParallel(model)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.01,
                       weight_decay=0.0001)
loss_function = F.cross_entropy

train_model(model, train_iter, optimizer, loss_function, n_gpus, 20, 'LSTM')
test_model(model, os.path.join(path.rsplit('/', 1), 'model_cache/LSTM.pt'),
           n_gpus, test_iter, loss_function)
