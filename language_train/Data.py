from torchtext.legacy.data import Iterator
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy import data
from torchtext.vocab import Vectors
from torch.nn import init
import torch
from tqdm import tqdm
import random
import jieba
import json
import os
import dill
import numpy as np
import sys

sys.path.append('..')
from utils import path

data_path = path.rsplit('/', 1)[0]


class PriceDataset(Dataset):
    def __init__(self, train_or_test='train'):
        super().__init__()
        x_data = np.load(os.path.join(data_path, 'data/x_price_data.npy'))
        if train_or_test == 'train':
            self.x = x_data[:974 * 1119]
        elif train_or_test == 'test':
            self.x = x_data[974 * 1119:]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


class MyTextDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.news)

    def __init__(self, examples, fields, filter_pred=None):
        super(MyTextDataset, self).__init__(examples, fields, filter_pred)


# 定义Dataset
class MyDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self,
                 file_path,
                 NEWS,
                 LAST_NEWS_PERIOD,
                 LABEL,
                 test_or_train='test',
                 aug=False,
                 **kwargs):

        fields = [('news', NEWS), ('news_period', LAST_NEWS_PERIOD),
                  ("label", LABEL)]

        #定义torchtext的example例子的列表
        examples = []

        print('read data from {}'.format(file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            news_content = json.load(f)

        #二分类
        lable_path = os.path.join(data_path, 'data/label_2.npy')
        label_all = np.load(lable_path)
        print('read lables from {}'.format(lable_path))
        #三分类
        #label_all = np.load(os.path.join(data_path,'data/label_3.npy'))
        print('开始处理example...')
        if test_or_train == 'test':
            labels = label_all[974:]
            labels = labels.reshape(-1, 1).squeeze()
            news_content = news_content[:-1119]
            print('processing...')
            for text, label in tqdm(zip(news_content, labels)):
                #为了不使分类时，发生onehot编码错误，这里将下降的-1改为2。
                if label == -1:
                    label = 2
                examples.append(
                    data.Example.fromlist([
                        ' '.join(text['news']), text['last_news_period'], label
                    ], fields))
        elif test_or_train == 'train':
            labels = label_all[:974]
            labels = labels.reshape(-1, 1).squeeze()  #974
            print('processing...')
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
        else:
            labels = np.random.randint(2, size=len(news_content))  #限定二分类
            for text, label in tqdm(zip(news_content, labels)):
                examples.append(
                    data.Example.fromlist([
                        ' '.join(text['news']), text['last_news_period'], label
                    ], fields))
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


def get_field():
    #获取停词表
    stop_words = stop_word()
    #定义field
    NEWS = data.Field(
        sequential=True,
        batch_first=True,
        tokenize=tokenizer,
        fix_length=300,  #在制作train_set,test_set,validation_set时，这里应该是1000
        stop_words=stop_words)
    LAST_NEWS_PERIOD = data.Field(sequential=False,
                                  use_vocab=False,
                                  preprocessing=lambda x: int(x),
                                  dtype=torch.int8)
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.int8)
    return NEWS, LAST_NEWS_PERIOD, LABEL


def stop_word():
    """获取停用词

    Returns:
        list: 停用词
    """
    #读取停词表
    stopword_path = os.path.join(data_path, 'data/chnstopwords.txt')
    stop_word = []
    with open(stopword_path, 'r', encoding='gbk') as f:
        for line in f:
            stop_word.append(line.strip())
    return stop_word


def tokenizer(text_):
    """中文分词

    Args:
        text_ (str): 中文文本

    Returns:
        list: 分词后的列表
    """
    #finance_dict = os.path.join(data_path, 'data/financedict.txt')
    #jieba.load_userdict(finance_dict)
    #首先进行split,因为存在空格
    text = text_.split()
    text_list = []
    for text_piece in text:
        seg = jieba.cut(text_piece, use_paddle=True)
        for word in seg:
            #4.去除停词以及去掉单独一个字的文本
            if len(word) > 1:
                text_list.append(word)
    return text_list


def get_train_set(NEWS, LAST_NEWS_PERIOD, LABEL):
    """获取训练集

    Returns:
        _type_: 训练集
    """
    print('train_set...')
    train = MyDataset(os.path.join(data_path, 'data/train_set.json'),
                      NEWS=NEWS,
                      LAST_NEWS_PERIOD=LAST_NEWS_PERIOD,
                      LABEL=LABEL,
                      test_or_train='train',
                      aug=0)
    return train


def get_test_set(NEWS, LAST_NEWS_PERIOD, LABEL):
    """获取测试集

    Returns:
        _type_: 测试集
    """
    print('test_set...')
    test = MyDataset(os.path.join(data_path, 'data/test_set.json'),
                     NEWS=NEWS,
                     LAST_NEWS_PERIOD=LAST_NEWS_PERIOD,
                     LABEL=LABEL,
                     test_or_train='test',
                     aug=0)
    return test


def get_random_set(NEWS, LAST_NEWS_PERIOD, LABEL):
    """随机获取一个数据集

    Returns:
        _type_: _description_
    """
    print('test_set...')
    random_set = MyDataset(
        os.path.join(data_path, 'data/1.json'),  #1可以改
        NEWS=NEWS,
        LAST_NEWS_PERIOD=LAST_NEWS_PERIOD,
        LABEL=LABEL,
        test_or_train='test',
        aug=0)
    return random_set


def get_pre_vector():
    """获取预训练的词向量

    Returns:
        _type_: 词向量
    """
    #加载中文词向量，链接见https://github.com/Embedding/Chinese-Word-Vectors
    vectors = Vectors(name=os.path.join(
        data_path,
        'chinese_word_embedding/sgns.financial.word/sgns.financial.word'),
                      cache='.vector_cache',
                      unk_init=init.xavier_normal_)  #改了
    return vectors


def get_vocab():
    """创建词表，主要是为了之后加快数据读取
    """
    NEWS, LAST_NEWS_PERIOD, LABEL = get_field()
    train = get_train_set(NEWS, LAST_NEWS_PERIOD, LABEL)
    test = get_test_set(NEWS, LAST_NEWS_PERIOD, LABEL)
    vec = get_pre_vector()
    #构建词表
    NEWS.build_vocab(train, test, vectors=vec, min_freq=5)
    NEWS.vocab.vectors.unk_init = init.xavier_uniform
    #保持词表
    if os.path.exists('/new_python_for_gnn/毕设code/data/text_vocab'):
        print('已经存在词表')
    else:
        with open('/new_python_for_gnn/毕设code/data/text_vocab', 'wb') as f:
            dill.dump(NEWS.vocab, f)
        print('完成词表的创建')
    #保存数据集
    if os.path.exists(
            '/new_python_for_gnn/毕设code/data/trian_set') and os.path.exists(
                '/new_python_for_gnn/毕设code/data/test_set'):
        print('训练集和测试集已经有保存')
    else:
        with open('/new_python_for_gnn/毕设code/data/trian_set', 'wb') as f:
            dill.dump(train.examples, f)
        print('已保存训练集')
        with open('/new_python_for_gnn/毕设code/data/test_set', 'wb') as f:
            dill.dump(test.examples, f)
        print('已保存测试集')


if __name__ == '__main__':
    #get_vocab()
    NEWS, LAST_NEWS_PERIOD, LABEL = get_field()
    print('读取数据')
    train = get_train_set(NEWS, LAST_NEWS_PERIOD, LABEL)
    print('读取数据完成')
    print('存储...')
    ##事实上训练集太大了，分出来一部分做验证集，前774天为训练集，后200天为验证集
    with open('/new_python_for_gnn/毕设code/data/trian_set', 'wb') as f:
        dill.dump(train.examples[:1119 * 774], f)
    print('训练集保存完成')
    with open('/new_python_for_gnn/毕设code/data/validation_set', 'wb') as f1:
        dill.dump(train.examples[1119 * 774:], f1)
    print('验证集保存完成')