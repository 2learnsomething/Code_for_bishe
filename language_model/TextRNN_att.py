import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 hidden_size,
                 hidden_size2,
                 num_layers,
                 num_classes,
                 weight_matrix,
                 drop_out=0.5):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=drop_out)
        self.tanh1 = nn.Tanh()
        #self.u = nn.Parameter(torch.Tensor(hidden_size * 2,hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2 + 4, num_classes)
        self.value = torch.tensor(1.0, requires_grad=True, dtype=float)

    def forward(self, x, period, price):
        emb = self.embedding(x)
        period_ = torch.exp(self.value * period / 30)
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        #M = torch.tanh(torch.matmul(H, self.u))  #可以注释掉
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = out.mean(1)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period_)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = F.tanh(out)
        out = self.fc1(out)
        out = F.tanh(out)
        out = torch.cat((out, price), dim=1)
        out = self.fc(out)
        return out