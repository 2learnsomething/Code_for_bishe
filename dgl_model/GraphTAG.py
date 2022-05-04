from cmath import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs
import dgl
from dgl.nn.pytorch import GraphConv, TAGConv


class CNN_layer(nn.Module):
    def __init__(self, n_vocab, embed_dim, hidden_size, num_filters,
                 filter_size, weight_matrix):

        super(CNN_layer, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_size])
        self.fc = nn.Linear(num_filters * len(filter_size), hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
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
        out = self.fc(out)
        out = F.tanh(out)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.div(out, period)
        out = torch.transpose(out, dim0=0, dim1=1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = torch.cat((out, price), 1)
        return out


class Model(nn.Module):
    def __init__(self, graph_type, n_vocab, embed_dim, hidden_size,
                 num_filters, filter_size, weight_matrix, num_classes):
        super(Model, self).__init__()
        self.graph_type = graph_type
        self.g = self.graph_make()
        self.g = dgl.add_self_loop(self.g)
        self.g = self.g.to(torch.device('cuda:0'))
        self.cnn = CNN_layer(n_vocab, embed_dim, hidden_size, num_filters,
                             filter_size, weight_matrix)
        self.conv1 = GraphConv(hidden_size + 4,
                               hidden_size,
                               allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size,
                               hidden_size,
                               allow_zero_in_degree=True)
        self.tag = TAGConv(hidden_size, hidden_size, k=3)
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
        out = self.tag(self.g, out)
        #out = out.mean(1)
        out = F.tanh(out)
        out = self.fc(out)
        return out