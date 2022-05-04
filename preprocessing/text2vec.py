import torch.nn.functional as F
import torch.optim as optim
import torch
import sys 
sys.path.append('.')
from language_train.Dataset import DEVICE, test_model, train_model, dataset
from langugade_model.LSTM import LSTM

print(DEVICE)
train_iter,test_iter,NEWS = dataset('both')
model = LSTM(len(NEWS.vocab),hidden_size=128,num_layers=2,classes=2)
weight_matrix = NEWS.vocab.vectors
model.embedding.weight.data.copy_(weight_matrix)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
loss_function = F.cross_entropy

train_model(model,train_iter,optimizer,loss_function,DEVICE,1,'LSTM')
test_model(model,'D:\毕设code\model_cache\LSTM.pt',test_iter,loss_function,DEVICE)
