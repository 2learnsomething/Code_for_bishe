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
from language_model.FastText import Model
from utils import path

import warnings

warnings.filterwarnings("ignore")

path = path.rsplit('/', 1)[0]


def main():
    print('   ======Begin======   ')
    print('开始时间:', time.ctime())
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
    # 进一步处理两个数据集，因为发现有部分数据仍然是空，这就比较麻烦,
    # 这里只能极端一点直接补充内容为'公司名称'+'股价'+'上升'/'下降'
    # 因为完成毕设的时间快不够了，呜呜呜呜呜呜呜呜呜呜呜呜呜呜呜。
    #print('processing train and test further...')
    #train_set, test_set = company_deal(train_set, test_set)
    #特征选择
    # print('feature selection...')
    # time9 = time.time()
    # train_set, test_set = company_deal(train_set, test_set)
    # time10 = time.time()
    # print('feature selection time:', time10 - time9)
    #获取预训练词向量
    print('receiving pretrained vectors...')
    vectors = get_pre_vector(
    )  #注Vectors源文件做了修改，见https://blog.csdn.net/qq_23262411/article/details/100173224
    #定义词表
    print('define vocab...')
    time3 = time.time()
    NEWS.build_vocab(
        train_set, test_set, vectors=vectors
    )  #去除频率小于十次的词,这会导致一部分资讯没有内容，因此后来不保留这个min_fre(最一开始有的），也能留下比较好的特征
    #NEWS.vocab.vectors.unk_init = init.xavier_uniform_
    weight_vector = NEWS.vocab.vectors
    time4 = time.time()
    print('building vocab time:', time4 - time3)
    print('矩阵的维度为', weight_vector.shape)
    #print('矩阵的前十行为', weight_vector[:10])
    #生成文本迭代器
    print('define iterators...')
    BATCH_SIZE = 4096
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
    hidden_size = 256
    num_classes = 2
    weight_matrix = weight_vector
    model_name = 'FastText'
    n_epochs = 100  #先小点做测试
    num_iter = 5
    lr_ = 0.04
    lr_decay_ = 0.0125
    weight_decay_ = 0.0004
    model_path = '/new_python_for_gnn/毕设code/model_cache'
    #模型实例化
    print('define model...')
    fasttext = Model(n_vocab=n_vocab,
                     embed_dims=embed_dims,
                     hidden_size=hidden_size,
                     num_classes=num_classes,
                     weight_matrix=weight_matrix)
    print('模型结构:')
    print(fasttext)
    parameters_num = get_parameter_number(fasttext)
    print('模型总参数量为:')
    print('total:{},size:{:.3f}M'.format(parameters_num['Total'],
                                         parameters_num['Total'] / 1024**2))
    print('模型可训练参数量为:')
    print('train:{},size:{:.3f}M'.format(
        parameters_num['Trainable'], parameters_num['Trainable'] / 1024**2))
    #定义训练部件
    print('define optimizer and criterion...')
    optimizer = optim.Adam(fasttext.parameters(),
                           lr=lr_,
                           weight_decay=weight_decay_)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=1 - lr_decay_,
                                                     patience=5)
    criterion = F.cross_entropy
    #开始训练
    print('begin training...')
    time5 = time.time()
    train_model(fasttext, train_iter, train_price_iter, optimizer, criterion,
                scheduler, n_epochs, model_name, num_iter)
    time6 = time.time()
    print('training time:', time6 - time5)
    #开始测试
    print('begin tesing...')
    time7 = time.time()
    test_model(fasttext, os.path.join(model_path, model_name + '.pt'),
               test_iter, test_price_iter, model_name, num_iter)
    time8 = time.time()
    print('testing time:', time8 - time7)
    #全部完成
    print('   ======all done======   ')
    time2 = time.time()
    print('total time:', time2 - time1)
    print('结束时间:', time.ctime())


if __name__ == '__main__':
    main()
