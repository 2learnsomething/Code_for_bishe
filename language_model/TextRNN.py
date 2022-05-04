import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 hidden_size,
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
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1 = nn.Linear(hidden_size + 4, num_classes)
        self.value = torch.tensor(1.0, requires_grad=True, dtype=float)

    def forward(self, x, period, price):
        out = self.embedding(x)
        period_ = torch.exp(self.value * period / 30)
        out, _ = self.lstm(out)
        out = out.mean(1)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period_)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc(out)
        out = F.tanh(out)
        out = torch.cat((out, price), dim=1)
        out = self.fc1(out)
        #out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out