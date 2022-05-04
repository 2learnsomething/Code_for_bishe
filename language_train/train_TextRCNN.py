import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torchtext.legacy.data import Iterator
from torch.utils.data import DataLoader
import dill
import os
import time
import sys

sys.path.append('.')
from Data import MyTextDataset, get_field, get_pre_vector, PriceDataset
from utils_dl import train_model, test_model, get_parameter_number, company_deal
from language_model.TextRCNN import Model
from utils import path

import warnings

warnings.filterwarnings("ignore")

path = path.rsplit('/', 1)[0]


def main():
    print('   ======Begin======   ')
    print('start time:', time.ctime())
    time1 = time.time()
    #获取数据集路径
    print('receiving data path...')
    train_data = os.path.join(path, 'data', 'train_set_new')
    validation_data = os.path.join(path, 'data', 'validation_set_new')
    test_data = os.path.join(path, 'data', 'test_set_new')
    #读取数据
    print('reading data...')
    start_time = time.time()
    with open(train_data, 'rb') as f:
        train_example = dill.load(f)
    with open(validation_data, 'rb') as f:
        validation_example = dill.load(f)
    with open(test_data, 'rb') as f:
        test_example = dill.load(f)
    end_time = time.time()
    print('data reading time:', end_time - start_time)
    #定义字段
    print('defining fields...')
    NEWS, LAST_NEWS_PERIOD, LABEL = get_field()
    fields = [('news', NEWS), ('news_period', LAST_NEWS_PERIOD),
              ("label", LABEL)]
    #构建数据,不单独区分测试集了
    print('building dataset...')
    train_set = MyTextDataset(examples=train_example + validation_example,
                              fields=fields)
    test_set = MyTextDataset(examples=test_example, fields=fields)
    #获取预训练词向量
    print('receiving pretrained vectors...')
    vectors = get_pre_vector(
    )  #注Vectors源文件做了修改，见https://blog.csdn.net/qq_23262411/article/details/100173224
    #定义词表
    print('define vocab...')
    time3 = time.time()
    NEWS.build_vocab(train_set, test_set, vectors=vectors)
    weight_vector = NEWS.vocab.vectors
    time4 = time.time()
    print('building vocab time:', time4 - time3)
    print('矩阵的维度为', weight_vector.shape)
    #生成文本迭代器
    print('define iterators...')
    BATCH_SIZE = 972
    train_iter = Iterator(train_set,
                          batch_size=BATCH_SIZE,
                          device=-1,
                          sort_key=lambda x: len(x.news),
                          sort_within_batch=False,
                          shuffle=False,
                          repeat=False)
    test_iter = Iterator(test_set,
                         batch_size=BATCH_SIZE,
                         device=-1,
                         sort=False,
                         sort_within_batch=False,
                         shuffle=False,
                         repeat=False)
    #构建股价数据集
    print('price dataset...')
    train_price = PriceDataset(train_or_test='train')
    test_price = PriceDataset(train_or_test='test')
    #构建迭代器
    print('price iterators...')
    train_price_iter = DataLoader(train_price,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    test_price_iter = DataLoader(test_price,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    #定义模型参数
    print('define model parameter...')
    n_vocab = len(NEWS.vocab)
    embed_dims = 300
    hidden_size = 128
    num_layers = 5  #10
    num_classes = 2
    pad_size = 16
    weight_matrix = weight_vector
    model_name = 'TextRCNN'
    n_epochs = 100  #先小点做测试
    num_iter = 200
    lr_ = 0.01
    lr_decay_ = 0.0125
    weight_decay_ = 0.0005
    is_clip = True
    train_again = False
    model_path = '/new_python_for_gnn/毕设code/model_cache'
    #模型实例化
    print('define model...')
    textrcnn = Model(n_vocab=n_vocab,
                     embed_dim=embed_dims,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     pad_size=pad_size,
                     num_classes=num_classes,
                     weight_matrix=weight_matrix)
    print('模型结构:')
    print(textrcnn)
    parameters_num = get_parameter_number(textrcnn)
    print('total parameter:')
    print('total:{},size:{:.3f}M'.format(parameters_num['Total'],
                                         parameters_num['Total'] / 1024**2))
    print('trainable parameter:')
    print('train:{},size:{:.3f}M'.format(
        parameters_num['Trainable'], parameters_num['Trainable'] / 1024**2))
    #定义训练部件
    print('define optimizer and criterion...')
    optimizer = optim.Adam(textrcnn.parameters(),
                           lr=lr_,
                           weight_decay=weight_decay_,
                           eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=1 - lr_decay_,
                                                     patience=1)
    criterion = F.cross_entropy
    #开始训练
    print('begin training...')
    time5 = time.time()
    train_model(textrcnn, train_iter, train_price_iter, optimizer, criterion,
                scheduler, n_epochs, model_name, num_iter, is_clip,
                train_again)
    time6 = time.time()
    print('training time:', time6 - time5)
    #开始测试
    print('begin tesing...')
    time7 = time.time()
    test_model(textrcnn, os.path.join(model_path, model_name + '.pt'),
               test_iter, test_price_iter, model_name, num_iter)
    time8 = time.time()
    print('testing time:', time8 - time7)
    #全部完成
    print('   ======all done======   ')
    time2 = time.time()
    print('total time:', time2 - time1)
    print('end time:', time.ctime())


if __name__ == '__main__':
    main()
