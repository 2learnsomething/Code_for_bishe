import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 pad_size,
                 num_classes,
                 weight_matrix,
                 drop_out=0.5):
        """_summary_

        Args:
            n_vocab (_type_): 词表大小
            embed_dim (_type_): 嵌入维度
            hidden_size (_type_): 隐藏层尺寸
            num_layers (_type_): 层数
            pad_size (_type_): _description_
            num_classes (_type_): 种类数
            weight_matrix (_type_): 预训练权重
            drop_out (float, optional): _description_. Defaults to 0.5.
        """
        super(Model, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=drop_out)
        self.maxpool = nn.MaxPool1d(pad_size, stride=2)
        self.fc = nn.Linear(556, hidden_size)
        self.value = torch.tensor(1.0, requires_grad=True, dtype=float)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + 4, num_classes)

    def forward(self, x, period, price):
        embed = self.embedding(x)
        period_ = torch.exp(self.value * period / 30)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)  #556
        out = F.tanh(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = out.permute(0, 2, 1)
        out = self.fc(out)
        out = F.tanh(out)
        out = out.mean(1)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period_)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = torch.cat((out, price), dim=1)
        out = self.fc2(out)
        return out


# if __name__ == '__main__':
#     input = torch.randn((128, 1000, 300))
#     lstm = nn.LSTM(300,
#                    128,
#                    2,
#                    bidirectional=True,
#                    batch_first=True,
#                    dropout=0.5)
#     maxpool = nn.MaxPool1d(16, stride=2)
#     out, _ = lstm(input)
#     out = torch.cat((input, out), 2)  #556
#     out = F.tanh(out)
#     out = out.permute(0, 2, 1)
#     out = maxpool(out).squeeze()
#     out = out.permute(0, 2, 1)
#     #print(out.shape)  #[128,62,556] ; strid=2 ,[128,493,556]
#     fc = nn.Linear(556, 128)
#     out = fc(out)  #[128,493,128]
#     out = out.mean(1)
#     #print(out.shape) #[128,128]
#     period_ = torch.randn(128)
#     out = torch.transpose(out, dim0=0, dim1=1)
#     out = torch.div(out, period_)
#     out = torch.transpose(out, dim0=0, dim1=1)
#     print(out.shape)
