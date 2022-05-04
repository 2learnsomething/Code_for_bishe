import json
import os
import sys

sys.path.append('..')
from preprocessing.price_preprocess import new_left_company
from utils import path

path = path.rsplit("/", 1)[0]
print(path)

file_list = new_left_company(os.path.join(path, 'data'))
file_use = []
for file_name in file_list:
    if file_name.find('.json') != -1:
        file_use.append(file_name)
print('get all data file name...')
file_use.sort(key=lambda x: int(x.replace('.json', '')))
print(file_use)
print('data_file num', len(file_use))

#训练集
train_set = []
for index, file_1 in enumerate(file_use[:9]):
    print(f'reading {index}...')
    with open(os.path.join(path, 'data', file_1), 'r') as f1:
        json_content = json.load(f1)
        train_set += json_content[::-1]
    print('read...')

with open(os.path.join(path, 'data', 'train.json'), 'wb') as f:
    data_ = json.dumps(train_set, indent=4,
                       ensure_ascii=False).encode(encoding='utf-8')
    f.write(data_)
print('finished train set...')
print('train size:', len(train_set) / 1119)

#测试集
test_set = []
for index, file_2 in enumerate(file_use[9:]):
    print(f'reading {index}...')
    with open(os.path.join(path, 'data', file_2), 'r') as f2:
        json_content = json.load(f2)
        test_set += json_content[::-1]
    print('read...')

with open(os.path.join(path, 'data', 'test.json'), 'wb') as f:
    data__ = json.dumps(test_set, indent=4,
                        ensure_ascii=False).encode(encoding='utf-8')
    f.write(data__)
print('finished test set...')
print('test size:', len(test_set) / 1119)
