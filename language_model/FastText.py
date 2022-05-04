import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 n_vocab,
                 embed_dims,
                 hidden_size,
                 num_classes,
                 weight_matrix,
                 drop_out=0.1):
        """

        Args:
            n_vocab (_type_): 词表大小
            embed_dims (_type_): 词嵌入维度
            hidden_size (_type_): 隐藏层大小
            num_classes (_type_): 分类数
            weight_matrix (_type_): 预训练权重
            drop_out (float, optional):  Defaults to 0.5.
        """
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dims)
        self.embedding.weight.data.copy_(weight_matrix)
        #self.embedding2 = nn.Embedding(n_vocab, embed_dims)
        #self.embedding3 = nn.Embedding(n_vocab, embed_dims)
        self.fc = nn.Linear(embed_dims, embed_dims // 4)
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(embed_dims // 4, hidden_size)
        self.dropout2 = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(drop_out)
        self.fc3 = nn.Linear(hidden_size + 4, num_classes)

    def forward(self, x, period, price):

        out = self.embedding(x)  # [4096/4，600，300]
        period_ = torch.exp(period / 30)  #一个月的期限,[4096/4，4]
        #out = torch.transpose(out, dim0=1, dim1=2)
        #out = torch.transpose(out, dim0=1, dim1=2)
        #out2 = self.embedding2(x)
        #out3 = self.embedding3(x)
        #out = torch.cat((out1, out2, out3), -1)
        out = self.fc(out)
        out = F.tanh(out)
        out = out.mean(dim=1)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period_)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.tanh(out)
        out = self.dropout3(out)
        out = torch.cat((out, price), dim=1)
        out = self.fc3(out)
        return out