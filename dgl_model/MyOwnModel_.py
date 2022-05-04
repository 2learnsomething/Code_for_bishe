from cmath import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs
import dgl
from dgl.nn.pytorch import GraphConv, GATConv


class CNN_layer(nn.Module):
    def __init__(self, n_vocab, embed_dim, hidden_size, num_layers,
                 num_filters, filter_size, weight_matrix):

        super(CNN_layer, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        #长距离特征
        self.lstm = nn.LSTM(embed_dim,
                            hidden_size,
                            num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.5)
        #局部信息
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_size])
        self.fc = nn.Linear(num_filters * len(filter_size), hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.value = torch.tensor(1.0, dtype=float, requires_grad=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input):
        x, period, price = input
        out = self.embedding(x)
        out, _ = self.lstm(out)  #[batch_size,seq_len,direction*hidden_size]
        period = torch.exp(self.value * period / 30)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs],
                        1)
        out = self.fc(out)
        out = F.tanh(out)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = torch.cat((out, price), 1)
        return out


class Graph_model(nn.Module):
    def __init__(self, graph_type, n_vocab, embed_dim, hidden_size,
                 num_filters, filter_size, num_layers, num_heads,
                 weight_matrix, num_classes):
        super(Graph_model, self).__init__()
        self.graph_type = graph_type
        self.g = self.graph_make()
        self.g = dgl.add_self_loop(self.g)
        self.g = self.g.to(torch.device('cuda:0'))
        self.cnn = CNN_layer(n_vocab, embed_dim, hidden_size, num_filters,
                             num_layers, filter_size, weight_matrix)
        self.conv1 = GraphConv(hidden_size + 4,
                               hidden_size,
                               allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size,
                               hidden_size,
                               allow_zero_in_degree=True)
        self.gat = GATConv(hidden_size, hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, num_classes)

    def graph_make(self):
        glist, lable = load_graphs('/new_python_for_gnn/毕设code/data/graph.bin')
        return glist[self.graph_type]

    def forward(self, x, period, price):
        out = self.cnn(x, period, price)
        out = self.conv1(self.g, out)
        out = F.tanh(out)
        out = self.conv2(self.g, out)
        out = F.tanh(out)
        out = self.gat(self.g, out)
        out = out.mean(1)
        out = F.tanh(out)
        out = self.fc(out)
        return out


def conv_and_pool(x, conv):
    x = F.relu(conv(x)).squeeze(3)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x


if __name__ == '__main__':
    embed_dims = 300
    hidden_size = 128
    num_layers = 2
    num_filters = 256
    filter_size = range(1, 6)
    num_heads = 5
    num_classes = 2
    model_name = 'MyOwnModel'
    n_epochs = 1  #先小点做测试
    num_iter = 50
    lr_ = 0.01
    lr_decay_ = 0.0125
    weight_decay_ = 0.0005
    is_clip = True
    model_path = '/new_python_for_gnn/毕设code/model_cache'

    convs = nn.ModuleList(
        [nn.Conv2d(1, num_filters, (k, embed_dims)) for k in filter_size])

    x = torch.randn((1119, 1000, 300))
    lstm = nn.LSTM(300,
                   128,
                   3,
                   bidirectional=True,
                   batch_first=True,
                   dropout=0.5)
    out, _ = lstm(x)
    out = out.unsqueeze(1)
    out = torch.cat([conv_and_pool(out, conv) for conv in convs], 1)
    print(out.shape)
