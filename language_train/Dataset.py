from torchtext.legacy.data import Iterator
from torchtext.legacy import data
from torchtext.vocab import Vectors
from torch.nn import init
import torch
from tqdm import tqdm
import random
import jieba
import json
import os
import numpy as np
import sys

sys.path.append('.')
from utils import path as data_path


# 定义Dataset
class MyDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self,
                 path,
                 news_field,
                 news_period,
                 label_field,
                 test=False,
                 aug=False,
                 **kwargs):
        fields = [('news', news_field), ('news_period', news_period),
                  ("label", label_field)]

        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            news_content = json.load(f)
        print('read data from {}'.format(path))

        #二分类
        lable_path = os.path.join(
            data_path.rsplit('/', 1)[0], 'data/label_2.npy')
        label_all = np.load(lable_path)
        print('read lables from {}'.format(lable_path))
        #三分类
        #label_all = np.load(os.path.join(data_path.rsplit('/',1)[0],'data/label_3.npy'))

        #注：在 一开始翻转日期的时候，公司顺序也改变了，所以label下面进行了flip操作，这个之后在做图网络的时候需要改。
        if test:
            labels = label_all[974:]
            labels = np.flip(labels, axis=1)
            labels = labels.reshape(-1, 1).squeeze()
            news_content = news_content[:-1119]
            for text, label in tqdm(zip(news_content, labels)):
                #为了不使分类时，发生onehot编码错误，这里将下降的-1改为2。
                if label == -1:
                    label = 2
                examples.append(
                    data.Example.fromlist([
                        ' '.join(text['news']), text['last_news_period'], label
                    ], fields))
        else:
            labels = label_all[:974]
            labels = np.flip(labels, axis=1)
            labels = labels.reshape(-1, 1).squeeze()  #974
            for text, label in tqdm(zip(news_content, labels)):
                if label == -1:
                    label = 2
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text['news'])
                    else:
                        text = self.shuffle(text['news'])
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(
                    data.Example.fromlist([
                        ' '.join(text['news']), text['last_news_period'], label
                    ], fields))
        # 之前是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        # super(MyDataset, self).__init__(examples, fields, **kwargs)
        super(MyDataset, self).__init__(examples, fields)

    def shuffle(self, text):
        text = np.random.permutation(list(jieba.cut(text, use_paddle=True)))
        return ''.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = list(jieba.cut(text, use_paddle=True))
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ''.join(text)


def tokenizer(text_):
    """中文分词

    Args:
        text_ (str): 字符串

    Returns:
        list: 分词后的列表
    """
    text = text_.split()
    finance_dict = os.path.join(
        data_path.rsplit('/', 1)[0], 'data/financedict.txt')
    jieba.load_userdict(finance_dict)
    #3.分词
    text_list = []
    for text_piece in text:
        seg = jieba.cut(text_piece, use_paddle=True)
        for word in seg:
            #4.去除停词以及去掉单独一个字的文本
            if len(word) > 1:
                text_list.append(word)
    return text_list


def TEXT_LABEL():
    """基于训练集得到标签等field

    Args:
        train (Mydataset): 训练数据集

    Returns:
        tuple: 文本和标签,以及词向量
    """
    #text和label用于数据格式的构建
    #处理field
    NEWS = data.Field(
        sequential=True,
        tokenize=tokenizer,
        batch_first=True,
        fix_length=1000,
    )
    LAST_NEWS_PERIOD = data.Field(sequential=False,
                                  use_vocab=False,
                                  preprocessing=lambda x: int(x),
                                  dtype=torch.int8)
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.int8)
    #加载中文词向量，链接见https://github.com/Embedding/Chinese-Word-Vectors
    vectors = Vectors(name=os.path.join(
        data_path.rsplit('/', 1)[0],
        'chinese_word_embedding/sgns.financial.word/sgns.financial.word'),
                      cache='.vector_cache')
    return NEWS, LAST_NEWS_PERIOD, LABEL, vectors


def dataset(train_test):
    """返回训练集和测试集

    Args:
        train_test (str): 返回训练集还是测试集，还是二者

    Returns:
        tuple: 数据集和NEWS
    """
    NEWS, LAST_NEWS_PERIOD, LABEL, vectors = TEXT_LABEL()
    print('train_set...')
    train = MyDataset(os.path.join(
        data_path.rsplit('/', 1)[0], 'data/train.json'),
                      news_field=NEWS,
                      news_period=LAST_NEWS_PERIOD,
                      label_field=LABEL,
                      test=False,
                      aug=0)
    print('test_set...')
    test = MyDataset(os.path.join(
        data_path.rsplit('/', 1)[0], 'data/test.json'),
                     news_field=NEWS,
                     news_period=LAST_NEWS_PERIOD,
                     label_field=LABEL,
                     test=True,
                     aug=0)

    NEWS.build_vocab(train, test, vectors=vectors, min_freq=5)  #去除频率小于五次的词
    NEWS.vocab.vectors.unk_init = init.xavier_uniform
    train_set = Iterator(train,
                         batch_size=1119,
                         device=-1,
                         sort_key=lambda x: len(x.news),
                         sort_within_batch=False,
                         repeat=False)
    test_set = Iterator(test,
                        batch_size=1119,
                        device=-1,
                        sort=False,
                        sort_within_batch=False,
                        repeat=False)
    if train_test == 'train':
        return train_set, NEWS
    elif train_test == 'test':
        return test_set, NEWS
    elif train_test == 'all':
        return train_set, test_set, NEWS


def data_parallel(module, input, device_ids, output_device=None):
    """分布式计算

    Args:
        module (_type_): 模型
        input (_type_): 数据输入
        device_ids (list): 编号
        output_device (_type_, optional): 输出到的目标装置. Defaults to None.

    Returns:
        _type_: _description_
    """
    #https://blog.csdn.net/aiwanghuan5017/article/details/102147824
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = torch.nn.parallel.replicate(module, device_ids)
    print(f"replicas:{replicas}")

    inputs = torch.nn.parallel.scatter(input, device_ids)
    print(f"inputs:{type(inputs)}")
    for i in range(len(inputs)):
        print(f"input {i}:{inputs[i].shape}")

    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    print(f"outputs:{type(outputs)}")
    for i in range(len(outputs)):
        print(f"output {i}:{outputs[i].shape}")

    result = torch.nn.parallel.gather(outputs, output_device)
    return result
