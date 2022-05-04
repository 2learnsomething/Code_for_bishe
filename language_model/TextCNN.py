import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 hidden_size,
                 num_filters,
                 filter_size,
                 num_classes,
                 weight_matrix,
                 drop_out=0.5):
        """_summary_

        Args:
            n_vocab (_type_): 词表大小
            embed_dim (_type_): 隐藏层数
            num_filters (_type_): 过滤器数目
            filter_size (_type_): 过滤尺寸
            num_classes (_type_): 种类数
            weight_matrix (_type_): 预训练权重
            drop_out (float, optional): _description_. Defaults to 0.5.
        """
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_size])
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(num_filters * len(filter_size), hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + 4, num_classes)
        self.value = torch.tensor(1.0, dtype=float, requires_grad=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, period, price):
        out = self.embedding(x)
        period = torch.exp(self.value * period / 30)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs],
                        1)
        #out = self.dropout(out)
        out = self.fc(out)
        out = F.tanh(out)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = torch.cat((out, price), 1)
        out = self.fc2(out)
        return out
