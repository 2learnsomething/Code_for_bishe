import json
import os
from collections import OrderedDict
import random
import numpy as np
import gc
import torch
import sys
from datetime import datetime
from functools import reduce

sys.path.append('.')
from preprocessing.price_preprocess import new_left_company

#设置种子保证后面输出的一致性
np.random.seed(123456)
torch.seed()

Data_path = 'D:\\毕设code\\news_data_from2016to2021'
Year = ['2019']  # 用上一年的数据补全下一年，并且2016年不会用于之后的实验，只是为了补充数据,我的电脑只能一个一个处理


def news_fullfil(Year, Data_path, company_list):
    """新闻数据补充主体代码

    Args:
        Year (list): 年份列表
        Data_path (str): 父路径的字符串
        company_list (list): 公司列表
    """
    #得到数据路径
    func = lambda x: os.path.join(Data_path, x, x + '_data.json')
    news_path = list(map(func, Year))
    #先把所有咨询读取，然后放在一个列表中
    news_list = []
    for news in news_path:
        print('Reading data from: ' + news)
        with open(news, 'r') as f:
            content_ = json.load(f)
            news_list.append(content_)
    #最后的数据保存为列表来返回
    all_data = []
    #有多少家公司
    company_num = len(company_list)
    print('number of company：', company_num)
    print('Begining to fullfil missing news for some day...')
    for i in range(len(news_list)):
        news = news_list[i]
        print('preprocessing ' + news_path[i])
        #存储时间日期
        date = []
        #存储是否是交易日
        is_open = []
        #存储公司数据
        company_data = []
        for elem in news:
            # key, value = elem.items() 是不对的，会报错
            for key, value in elem.items():
                date.append(key)
                is_open.append(int(value['is_open']))  #注意字典里面是字符串
                company_data.append(value['company'])
        #将数据翻转过来，便于操作
        date = date[::-1]
        is_open = is_open[::-1]
        company_data = company_data[::-1]
        print(
            'finished dict_data classification and begining to fullfil missing data...'
        )
        #记录一下有多少天
        total_days = len(date)
        #第一段处理过程
        #先把数据处理为需要的格式
        for j in range(total_days):
            #先把所有的数据都处理一下
            for company in company_list:  #注意company_list是字符串的列表
                com_news = company_data[j].get(company)
                if com_news != None:
                    title = short_text_remove_title(
                        com_news['title'])  #这个先不动，事实上篇幅也不长，所以先不动
                    news_content = short_text_remove_content(
                        com_news['content'])  #直接进行一波处理
                    #接下来这一部分是为了数据格式
                    dict_data = OrderedDict()
                    dict_data['date'] = date[j]
                    dict_data['is_open'] = str(is_open[j])
                    dict_data['company'] = company
                    dict_data['title'] = title
                    dict_data['news'] = news_content
                    dict_data['last_news_period'] = 0  #目前先默认为0
                    dict_data['last_title_period'] = 0
                    all_data.append(dict_data)
                else:
                    print(company + ' data is not found and need debug...')
                    break
        print('finished the format of data processing...')
        # print('output one piece of data...')
        # print(all_data[random.randint(1, 10000)])

        #开始另一段处理过程
        print('begin another procedure...')
        all_data = no_tradeday_fullfil(all_data, is_open, company_num)
        print('preprocessed all...')

        #尝试一次能不能写入完，不能的话，就分三次，这个主要是针对2020和2021两年。
        # try:
        #     with open(os.path.join('data', Year[i] + '.json'), 'wb') as f:
        #         json_final_data = json.dumps(
        #             all_data, indent=4,
        #             ensure_ascii=False).encode(encoding='utf-8')
        #         f.write(json_final_data)
        #         print(f'完成{Year[i]}年文件写入')
        # except MemoryError as me:
        #     print(me)
        #     with open(os.path.join('data', Year[i] + '_1.json'), 'wb') as f:
        #         json_final_data = json.dumps(
        #             all_data[:len(all_data) // 3],
        #             indent=4,
        #             ensure_ascii=False).encode(encoding='utf-8')
        #         f.write(json_final_data)
        #         print(f'完成{Year[i]}年前一部分文件写入')
        #     with open(os.path.join('data', Year[i] + '_2.json'), 'wb') as f:
        #         json_final_data = json.dumps(
        #             all_data[len(all_data) // 3:-len(all_data) // 3],
        #             indent=4,
        #             ensure_ascii=False).encode(encoding='utf-8')
        #         f.write(json_final_data)
        #         print(f'完成{Year[i]}年中间文件写入')
        #     with open(os.path.join('data', Year[i] + '_3.json'), 'wb') as f:
        #         json_final_data = json.dumps(
        #             all_data[-len(all_data) // 3:],
        #             indent=4,
        #             ensure_ascii=False).encode(encoding='utf-8')
        #         f.write(json_final_data)
        #         print(f'完成{Year[i]}年后部分文件写入')
        #以下主要是针对17，18，19三年
        with open(os.path.join('data', Year[i] + '_1.json'), 'wb') as f:
                json_final_data = json.dumps(
                    all_data[:1135*180],
                    indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f.write(json_final_data)
                print(f'完成{Year[i]}年前一部分文件写入')
        with open(os.path.join('data', Year[i] + '_2.json'), 'wb') as f:
            json_final_data = json.dumps(
                all_data[1135*180:],
                indent=4,
                ensure_ascii=False).encode(encoding='utf-8')
            f.write(json_final_data)
            print(f'完成{Year[i]}年后一部分文件写入')


def no_tradeday_fullfil(all_data, is_open, company_num):
    """对于缺乏数据的日子进行补充

    Args:
        all_data (list): 所有数据的列表
        is_open (list): 是否交易日对应的列表，1代表交易日
        company_num (int): 公司的数量
    """
    # 首先做一下分析，第一天数据对应的index为零到company_num-1，依次往后
    #判断一下数据个数对不对
    if len(all_data) == len(is_open) * company_num:
        for index in range(len(is_open)):
            if not is_open[index]:
                index_list, is_deal = is_trade(index, is_open)
                index_1 = list(map(lambda x: x * company_num, index_list))
                index_2 = list(map(lambda x: (x + 1) * company_num,
                                   index_list))
                index_new = zip(index_1, index_2)
                print(index, index_new)
                if is_deal:
                    all_data = news_compliment_not(index_new, all_data)
            else:
                #通过去两周的数据
                index_list = range(index, index + min(30,
                                                      len(is_open) - index))
                index_1 = list(map(lambda x: x * company_num, index_list))
                index_2 = list(map(lambda x: (x + 1) * company_num,
                                   index_list))
                index_new = zip(index_1, index_2)
                print(index, index_new)
                all_data = news_compliment_yes(index_new, all_data)
    else:
        print('the number piece of data is wrong.')
    #返回数据
    return all_data


def is_trade(index_, is_open):
    """得到相互补充数据的相邻几天

    Args:
        index (int): 非交易日的第一个下标
        is_open (list): 交易日的列表

    Returns:
        list: 非交易日和最近的一个交易日的下标列表
    """
    index_list = []
    while index_ < len(is_open) and (not is_open[index_]):
        index_list.append(index_)
        index_ += 1
    if index_ < len(is_open):
        index_list.append(index_)
    return index_list, is_open[index_list[-1]]


#注这里还不能进行分词，因为考虑到数据格式为json作为torchtext的输入
def short_text_remove_content(text):
    """对文本列表删除长度小于3的内容,针对文本正文内容

    Args:
        text (list): 不同篇资讯的嵌套列表

    Returns:
        list: 去除之后的列表，分两种一种是区分文章，一种是合并文章，这里先采取合并文章的办法
    """
    #事实上有的时候，资讯不止一条，所以需要遍历一下
    new_text = []
    #前途是非空,并且不是嵌套了一个空的列表，len([[]])= 1
    if text and text != [[]] * len(text):
        #print('begining to remove short length sentence...')
        for index in range(len(text)):
            #还是得非空
            if text[index]:
                text_left = []
                for elem in text[index]:
                    if len(elem) > 3:  #排除类似'万亿元'这种的内容
                        text_left.append(elem)
                #这里先以将所有有的资讯全部合并到一起,变成一篇文章，之后能够处理相应的数据格式了再说可以改
                #next_text.append(text_left)
                new_text += text_left
            else:
                continue
        return new_text
    else:
        return []


def short_text_remove_title(text):
    """对文本列表删除长度小于3的内容,针对标题内容

    Args:
        text (list): 不同篇资讯的嵌套列表

    Returns:
        list: 去除之后的列表，分两种一种是区分文章，一种是合并文章，这里先采取合并文章的办法
    """
    #事实上有的时候，资讯不止一条，所以需要遍历一下
    new_text = []
    #前途是非空,并且不是嵌套了一个空的列表，len([[]])= 1
    if text and text != [[]] * len(text):
        #print('begining to remove short length sentence...')
        for index in range(len(text)):
            #还是得非空
            if text[index]:
                text_left = []
                for elem in text[index]:
                    if len(elem) > 3:  #排除类似'万亿元'这种的内容
                        text_left.append(elem)
                #这里是和正文内容唯一的区别
                new_text.append(text_left)
                #new_text += text_left
            else:
                continue
        return new_text
    else:
        return []


def news_compliment_not(index_new, all_data):
    """对交易日数据进行补充,主要是第一天是非交易日

    Args:
        index_new (zip): 下标对组成的zip对象
        all_data (list): 数据列表

    Returns:
        list: 处理后的列表
    """
    data_use = []
    index_pair = []
    for idx1, idx2 in index_new:
        index_pair.append((idx1, idx2))
        data_use.append(all_data[idx1:idx2])
    #补数据
    for data in data_use[:-1]:
        for index, company_ in enumerate(data):
            if is_sublist(data_use[-1][index]['title'], company_['title']):
                data_use[-1][index]['title'] += company_['title']
            if is_sublist(data_use[-1][index]['news'], company_['news']):
                data_use[-1][index]['news'] += company_['news']
    #数据替换
    all_data[index_pair[-1][0]:index_pair[-1][1]] = data_use[-1]
    return all_data


def news_compliment_yes(index_new, all_data):
    """对交易日数据进行补充,主要是第一天是交易日

    Args:
        index_new (zip): 下标对组成的zip对象
        all_data (list): 数据列表

    Returns:
        list: 处理后的列表
    """
    #先读数据
    data_use = []
    index_pair = []
    for idx1, idx2 in index_new:
        index_pair.append((idx1, idx2))
        data_use.append(all_data[idx1:idx2])
    #补数据
    time_func = lambda x: datetime.strptime(x, '%Y-%m-%d')
    for index, data in enumerate(data_use[0]):
        #判断某天是否有标题数据
        if not data['title']:
            for compliment_data in data_use[1:]:
                compliment_data_ = compliment_data[index]
                if is_sublist(data['title'], compliment_data_['title']
                              ) and compliment_data_['title'] != [[]] * len(
                                  compliment_data_['title']):
                    data['title'] += compliment_data_['title']
                    data['last_title_period'] = (
                        time_func(data['date']) -
                        time_func(compliment_data_['date'])).days
            #print(f'complimented news title of {index}_th company...')
        #判断某天是否有正文数据

        if not data['news']:
            for compliment_data in data_use[1:]:
                compliment_data_ = compliment_data[index]
                if is_sublist(data['news'], compliment_data_['news']):
                    data['news'] += compliment_data_['news']
                    data['last_news_period'] = (
                        time_func(data['date']) -
                        time_func(compliment_data_['date'])).days
            #print(f'complimented news content of {index}_th company...')
    #数据替换
    all_data[index_pair[0][0]:index_pair[0][1]] = data_use[0]
    return all_data


def final_processing(data_pair, data_path, company_num):
    """对年与年之间相邻的月份进行数据补充

    Args:
        data_pair (zip): 下标组成的zip
        data_path (str): 数据保存路径
        company_num (int): 公司数量
    """
    for elem in data_pair:
        #读数据
        print('begining to process...')
        with open(os.path.join(data_path, elem[0]), 'r') as f1:
            data_old = json.load(f1)
        with open(os.path.join(data_path, elem[1]), 'r') as f2:
            data_new = json.load(f2)
        print('had read two data...')
        #需要处理的数据,举个例子来说，其实就是2017年的年初第一个月和2016年的最后一个月进行处理
        all_data = data_new[-company_num * 30:] + data_old[:company_num * 30]
        #记录是否是交易日
        is_open = []
        for i in range(60):
            is_open.append(int(all_data[i * company_num]['is_open']))
        print('processing...')
        all_data_ = no_tradeday_fullfil(all_data, is_open, company_num)
        print('finished...')
        data_new[-company_num *
                 30:], data_old[:company_num *
                                30] = all_data_[:company_num *
                                                30], all_data_[-company_num *
                                                               30:]
        #再重新写入
        print('rewriting...')
        try:
            with open(os.path.join(data_path, elem[0]), 'wb') as f3:
                json_final_data = json.dumps(
                    data_old, indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f3.write(json_final_data)
        except MemoryError as me:
            with open(os.path.join(data_path, '1_' + elem[0]), 'wb') as f3:
                json_final_data = json.dumps(
                    data_old[:len(data_old // 2)],
                    indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f3.write(json_final_data)
            with open(os.path.join(data_path, '2_' + elem[0]), 'wb') as f3:
                json_final_data = json.dumps(
                    data_old[len(data_old) // 2:],
                    indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f3.write(json_final_data)
        print('Success !!!')
        del data_old
        gc.collect()
        try:
            with open(os.path.join(data_path, elem[1]), 'wb') as f4:
                json_final_data = json.dumps(
                    data_new, indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f4.write(json_final_data)
        except MemoryError as me:
            with open(os.path.join(data_path, '1_' + elem[1]), 'wb') as f4:
                json_final_data = json.dumps(
                    data_new[:len(data_new) // 2],
                    indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f4.write(json_final_data)
            with open(os.path.join(data_path, '2_' + elem[1]), 'wb') as f4:
                json_final_data = json.dumps(
                    data_new[len(data_new) // 2:],
                    indent=4,
                    ensure_ascii=False).encode(encoding='utf-8')
                f4.write(json_final_data)
        print('Done...')
        del data_new
        gc.collect()


def is_sublist(content1, content2):
    """判断是否存在子集合关系

    Args:
        content1 (list): 主列表
        content2 (list): 子列表

    Returns:
        bool: 是否子集关系
    """
    is_sub = []
    if content2:
        for elem in content2:
            if elem in content1:
                is_sub.append(1)
            else:
                is_sub.append(0)
        #设置一个求和函数
        sum = reduce(lambda x, y: x + y, is_sub)
        if sum != len(is_sub):
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    #第一步，文件内的补充
    company_list = new_left_company(
        'news_data_from2016to2021\\companies\\final_use')
    company_list = list(map(lambda x: x.replace('.csv', ''), company_list))
    # news_fullfil(Year, Data_path, company_list)
    #第二步，文件间的补充
    # data_news = new_left_company('data')[:4]
    # data_pair = zip(data_news[:-1], data_news[1:])
    data_news1 =['2020_3.json']
    data_news2 = ['2021_1.json']
    data_pair = zip(data_news1,data_news2)
    final_processing(data_pair, 'data', len(company_list))
