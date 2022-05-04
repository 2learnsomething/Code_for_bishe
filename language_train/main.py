import torch.nn.functional as F
import torch.optim as optim
import os
import sys

sys.path.append('.')
from language_train.Dataset import train_model, dataset
from language_model.LSTM import LSTM
from utils import path
import warnings

warnings.filterwarnings("ignore")


def train(train_iter, NEWS):

    #超参数设置
    n_epochs = 100  # 训练轮数
    model_name = 'LSTM'  # 模型名称
    n_vocab = len(NEWS.vocab)  # 词表大小，在运行时赋值
    hidden_size = 128  # 隐藏层大小
    num_layers = 2  # 层数
    classes = 2  # 分类数
    weight_matrix = NEWS.vocab.vectors  #预训练词向量
    file_ab_path = path.rsplit('/', 1)[0]  #文件夹的绝对路径

    model = LSTM(n_vocab, hidden_size, num_layers, classes)

    model.embedding.weight.data.copy_(weight_matrix)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = F.cross_entropy

    train_model(model, train_iter, optimizer, criterion, n_epochs, model_name)
    #test_model(model,os.path.join(file_ab_path,'model_cache/'+model_name+'.pt'),test_iter)


def main():
    train_iter, NEWS = dataset('train')
    train(train_iter, NEWS)


if __name__ == '__main__':
    main()