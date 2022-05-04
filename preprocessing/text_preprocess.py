from ast import Str
from genericpath import exists
import string
from tracemalloc import stop
from turtle import left
from matplotlib.pyplot import title
from matplotlib.style import use
import jieba
import json
from collections import OrderedDict
from datetime import datetime
from zhon import hanzi
import os
import time
import pandas as pd
import re
from price_preprocess import get_company_name_code, new_left_company

#新闻所在路径
news_path = 'D:\毕设code\\news_data_from2016to2021'
#年份列表
year_list = ['2016', '2017', '2018', '2019', '2020', '2021']
#文件名称
file_name = 'ER_NewsInfo.xlsx'
#需要额外保存的列名
columns_name = ['DeclareDate', 'Classify', 'Title', 'NewsContent', 'Symbol']
#交易日数据
trade_date_path = 'news_data_from2016to2021\companies\\trade_cal_clean.csv'
#中文停词
chnstopword = 'D:\chrome\Listed-company-news-crawl-and-text-analysis-master\src\Leorio\chnstopwords.txt'
#储存最后使用的公司的路径
company_final = 'D:\毕设code\\news_data_from2016to2021\companies\\final_use'
#见price_preprocess
company_path = 'D:\毕设code\\news_data_from2016to2021\companies\深交所A股列表_主板.xlsx'


def trade_day(trade_day_path):
    """获取交易日数据

    Args:
        trade_day_path (file_path): 去除没用信息之后的新交易日csv的路径

    Returns:
        [dataframe]: 交易日的dataframe
    """
    return pd.read_csv(trade_day_path)


def date_trans(date):
    """时间格式转换

    Args:
        date (str): 字符串'20210112'这种

    Returns:
        [time]: 规范化之后的时间格式,2021-01-12
    """
    date = pd.to_datetime(date)
    return date.strftime('%Y-%m-%d')


def company_list(tripule):
    """得到名称，代码和简称的列表

    Args:
        tripule (三元组): 具体见get_company_name_code函数

    Returns:
        [list]: 三者各自的列表的元组
    """
    company_name, company_code, company_short = [], [], []
    for elem in tripule:
        company_name.append(elem[0])
        company_code.append(elem[1])
        company_short.append(elem[2])

    return company_name, company_code, company_short


def get_news_data(year):
    """获取指定年份的新闻资讯数据

    Args:
        year (int): 年份

    Returns:
        [dataframe]: 新闻资讯数据的dataframe,列名也修改了，注意！
    """
    start_time = time.time()
    news_name = os.path.join(news_path, str(year), file_name)
    news_data = pd.read_excel(news_name)  #获取新闻资讯数据
    end_time = time.time()
    print(end_time - start_time)
    #在该函数处就要转换
    news_data['DeclareDate'] = news_data['DeclareDate'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'
                                                                     ))
    news_data = news_data.dropna(subset=['Title', 'NewsContent'])
    return news_data[columns_name]


def get_specific_date_news(date, news_data):
    """获取指定日期的全部新闻

    Args:
        date (date): 需要返回数据的日期
        news_data (dataframe): 新闻数据

    Returns:
        [dataframe]: 指定日期的全部资讯的dataframe
    """
    date = date_trans(date)
    return news_data[news_data['DeclareDate'] == date]


def remove_braceket(text):
    """返回清洗过后的标题

    Args:
        text (str): 经过split之后的句子

    Returns:
        [str]: 清洗过后的结果
    """
    #title = re.sub('^《[\u4e00-\u9fa5_a-zA-Z0-9]+》','',title) 法律条令
    text = re.sub('^([\u4e00-\u9fa5_a-zA-Z0-9]+)', '', text)
    text = re.sub('^【[\u4e00-\u9fa5_a-zA-Z0-9]+】', '', text)
    text = re.sub('^@[\u4e00-\u9fa5_a-zA-Z0-9]+', '', text)
    return text


def title_preprocess(title):
    """将清洗过后的标题进行拼接返回

    Args:
        title (str): 原标题

    Returns:
        [str]: 拼接后的完整标题
    """
    title_list = title.split()
    title_list = [
        remove_braceket(title_constitution)
        for title_constitution in title_list
    ]
    title_list = [
        format_str(title_constitution) for title_constitution in title_list
    ]
    return ''.join(title_list)


# 让文本只保留汉字
def is_chinese(uchar):
    """判断是否是汉字

    Args:
        uchar (str): 汉字字符串

    Returns:
        [bool]: 是或者不是
    """
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def format_str(content):
    """返回句子

    Args:
        content (str): 要处理的中文句子

    Returns:
        [str]: 取掉标点符号之后的句子
    """
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


#todo，目前发现主要是链接，之后可能会有其他操作
def content_preprocess(content):
    """去除链接

    Args:
        content (str): 中文句子

    Returns:
        [str]: 去掉链接之后的字符串
    """
    #去除链接
    content = re.sub('^([a-zA-z]+://[^\s]*)', '', content)
    #去掉（）
    content = re.sub('[(](.*?)[)]', '', content)
    content = re.sub('[（](.*?)[）]', '', content)
    return content


def split_sentence(paragraph):
    """基于zhon进行中文分句

    Args:
        paragraph (str): 段落的中文字符串

    Returns:
        [list]: 分句组成的list
    """
    sentence_set = re.findall(hanzi.sentence, paragraph)
    return sentence_set


def news_preprocess(content):
    """去除中文的标点符号

    Args:
        content (str): 中文字符串

    Returns:
        [list]: 去除标点符号之后的句子列表
    """
    content = content.rsplit('（', 1)[0]  #去掉'（文章来源：xxxx）'
    content = content.replace('\u3000', ' ')  #首先将所有的全角空格替换掉
    content_split_list = content.split()  #此处只是得到段落，还需要处理
    content_list = []
    for paragraph in content_split_list:
        paragraph = content_preprocess(
            paragraph)  #去除奇怪内容，比如链接和括号，这里主要是中文括号，无论括号内什么内容都去除
        sentence_set = split_sentence(paragraph)  #段落的句子集合，包含奇奇怪怪的比如emoji等符号
        sentence_set = list(map(filter_str, sentence_set))  #去除奇怪符号
        content_list += sentence_set  #这个地方事实上没有考虑段落这一关系，这个地方之后如有需要可以改
        #content_list.append(sentence_set) 即可，就是段落区分，这个之后的处理也需要一点变化，先待定。
    chinese_list = []  #一条资讯全部句子
    for sentence in content_list:
        chinese_list.append(format_str(sentence))
    return chinese_list


def fenci(datas):
    """中文分词，事实上jieba之后可能还不一定用，这个有待商榷！！！！！！！！！！！！！！！！！！！！

    Args:
        datas (str): 需要进行分词的str

    Returns:
        list: 分词之后的列表
    """
    cut_words = map(lambda s: list(jieba.cut(s)), datas)
    return list(cut_words)


def filter_str(desstr, restr=''):
    """过滤除中英文及数字以外的其他字符

    Args:
        desstr ([str]): 目标字符串
        restr (str, optional): 做替换的字符串. Defaults to ''.

    Returns:
        [str]: 清洗过后的字符串
    """
    res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)


def single_company_news(company_name, company_code, company_short, date,
                        news_data):
    """获取新闻标题，内容以及资讯分类等信息

    Args:
        company_name (str): 全称
        company_code (str): 代码
        company_short (str): 简称
        date (date): 需要的某天资讯 
        year (str): 某一年的资讯

    Returns:
        [tuple]: 需要的信息的元组
    """
    #获取指定日期全部新闻
    specific_date_news = get_specific_date_news(date, news_data)
    #获取最后一行的信息
    index_list = specific_date_news.index.tolist()
    if index_list:
        end_line = index_list[-1]
    else:
        end_line = 0
    #去除newscontent为空的数据行
    specific_date_news = specific_date_news.dropna(
        subset=['NewsContent', 'Title'])
    #对相关联股票代码这一列进行填充操作
    specific_date_news = specific_date_news.fillna({'Symbol': '000000'})
    #index重置，这个很有必要
    specific_date_news = specific_date_news.reset_index(drop=True)
    # title列和newscontent列可能出现数字类型，这个需要进行处理
    specific_date_news[['Title', 'NewsConten'
                        ]] = specific_date_news[['Title',
                                                 'NewsContent']].astype(str)
    news_title, news_content, news_classification = [], [], []
    exists_news = False
    #获取某公司的相关新闻
    for idx in range(len(specific_date_news)):
        if specific_date_news.loc[idx,'Title'].find(company_name) != -1 or \
            specific_date_news.loc[idx,'NewsContent'].find(company_name) != -1 or \
            specific_date_news.loc[idx,'Title'].find(company_code) != -1 or \
            specific_date_news.loc[idx,'NewsContent'].find(company_code) != -1 or \
            specific_date_news.loc[idx,'Title'].find(company_short) != -1 or \
            specific_date_news.loc[idx,'NewsContent'].find(company_short) != -1 or \
            company_code == specific_date_news.loc[idx,'Symbol']:
            # 注意一下symbol列的情况。
            title = title_preprocess(specific_date_news.loc[idx, 'Title'])
            content = news_preprocess(specific_date_news.loc[idx,
                                                             'NewsContent'])
            news_title.append(title)
            news_content.append(content)
            news_classification.append(specific_date_news.loc[idx, 'Classify'])
    if len(news_title) != 0:
        exists_news = True
    return exists_news, news_title, news_content, news_classification, end_line


def company_detail(company_name, company_code, company_short):
    """返回公司信息的字典

    Args:
        company_name (str): 名称
        company_code (str): 代码
        company_short (str): 简称

    Returns:
        dict: 公司信息的字典
    """
    company = OrderedDict()
    #company['name'] = company_name
    #company['code'] = company_code
    #company['short'] = company_short
    return company


def maintain_relevent_news_single_year(year, company_final, trade_date_path):
    """将数据进行分年份保存,一次性写入

    Args:
        year (int): 年份
        company_final (str): 路径
        trade_date_path (str): 路径
    """
    #事实上没法手工去判断每一条新闻是否和某一家公司或者某几家公司有关联，
    # 所以简单些，就通过资讯是否提到某公司，或者说该资讯是否是行业资讯，
    # 来判断是否该咨询有用
    tripule = get_company_name_code()
    company_name, company_code, company_short = [], [], []
    company_left = new_left_company(company_final)  #获取留下来的公式code
    company_left_code = list(map(lambda x: x.replace('.csv', ''),
                                 company_left))
    #获取到留下的公司的代码，简称等信息
    for elem in tripule:
        if elem[1] in company_left_code:
            company_name.append(elem[0])
            company_code.append(elem[1])
            company_short.append(elem[2])
    #获取交易日数据
    trade_day_ = trade_day(trade_date_path)
    trade_day_['year'] = trade_day_['cal_date'].apply(
        lambda x: str(x)[:4])  #获取年份
    trade_day_['date'] = trade_day_['cal_date'].apply(
        lambda x: date_trans(str(x)))
    trade_day_data = trade_day_[trade_day_['year'] == str(year)]  #获取指定年份的交易日数据
    trade_day_data = trade_day_data.reset_index(drop=True)  #index重置
    trade_day_data['is_open'] = trade_day_data['is_open'].apply(
        lambda x: str(x))
    total_dict = OrderedDict()
    news_data = get_news_data(year)
    next_start = 0
    #创建一个字典，格式为
    # 日期：开市与否：1 or 0
    #       公司代码： 公司名称(解决)
    #              公司代码（解决）
    #              公司简称（解决）
    #              是否有新闻：(解决)
    #              股价情况：上升还是下降，相比前一天(这个先忽略吧)
    #              资讯类别：列表，行业资讯，还是个股资讯或者其他情况（解决）
    #              资讯title：列表，因为可能有多条资讯（解决）
    #              资讯内容：列表，可能有多条资讯（解决）
    print('----开始----')
    for index in range(len(trade_day_data) - 1,
                       len(trade_day_data)):  # 先拿一天做实验试试
        news_dict = OrderedDict()
        is_open = trade_day_data.loc[index, 'is_open']  #获取该日期是否开市
        news_dict['is_open'] = is_open
        company_data = OrderedDict()
        new_data = news_data.iloc[next_start:, :]
        for i in range(len(company_code)):
            company_dict = company_detail(
                company_name[i], company_code[i],
                company_short[i])  #为了降低文件大小，这里就不再计入公司信息了
            exists_news, news_title, news_content, news_classification, end_line = single_company_news(
                company_name[i], company_code[i], company_short[i],
                trade_day_data.loc[index, 'date'], new_data)
            company_dict['exists_news'] = exists_news
            company_dict['title'] = news_title
            company_dict['content'] = news_content
            company_dict['classification'] = news_classification
            company_data[company_code[i]] = company_dict
        #但是好像没有明显加快
        #为了加快速度,每一个日期在用完后，就drop掉，这样可能会提高一些速度
        #news_data.drop(index=news_data.loc[news_data['DeclareDate'] ==
        #                                   trade_day_data.loc[index,
        #                                                      'date']].index)
        #index要进行重置
        #news_data = news_data.reset_index(drop=True)
        next_start = end_line
        news_dict['company'] = company_data
        total_dict[trade_day_data.loc[index, 'date']] = news_dict
        print('完成{}的处理！'.format(trade_day_data.loc[index, 'date']))

    json_str = json.dumps(total_dict, indent=4,
                          ensure_ascii=False).encode(encoding='utf-8')
    with open(
            os.path.join('news_data_from2016to2021\companies',
                         str(year) + '_data.json'), 'wb') as json_file:
        json_file.write(json_str)
    print('----完成----')


def maintain_relevent_news_single_year_a(year, company_final, trade_date_path):
    """将数据进行分年份保存,一次性写入

    Args:
        year (int): 年份
        company_final (str): 路径
        trade_date_path (str): 路径
    """
    #事实上没法手工去判断每一条新闻是否和某一家公司或者某几家公司有关联，
    # 所以简单些，就通过资讯是否提到某公司，或者说该资讯是否是行业资讯，
    # 来判断是否该咨询有用
    tripule = get_company_name_code()
    company_name, company_code, company_short = [], [], []
    company_left = new_left_company(company_final)  #获取留下来的公式code
    company_left_code = list(map(lambda x: x.replace('.csv', ''),
                                 company_left))
    #获取到留下的公司的代码，简称等信息
    for elem in tripule:
        if elem[1] in company_left_code:
            company_name.append(elem[0])
            company_code.append(elem[1])
            company_short.append(elem[2])
    #获取交易日数据
    trade_day_ = trade_day(trade_date_path)
    trade_day_['year'] = trade_day_['cal_date'].apply(
        lambda x: str(x)[:4])  #获取年份
    trade_day_['date'] = trade_day_['cal_date'].apply(
        lambda x: date_trans(str(x)))
    trade_day_data = trade_day_[trade_day_['year'] == str(year)]  #获取指定年份的交易日数据
    trade_day_data = trade_day_data.reset_index(drop=True)  #index重置
    trade_day_data['is_open'] = trade_day_data['is_open'].apply(
        lambda x: str(x))
    news_data = get_news_data(year)
    total_dict = OrderedDict()
    next_start = 0
    #创建一个字典，格式为
    # 日期：开市与否：1 or 0
    #       公司代码： 公司名称(解决)
    #              公司代码（解决）
    #              公司简称（解决）
    #              是否有新闻：(解决)
    #              股价情况：上升还是下降，相比前一天(这个先忽略吧)
    #              资讯类别：列表，行业资讯，还是个股资讯或者其他情况（解决）
    #              资讯title：列表，因为可能有多条资讯（解决）
    #              资讯内容：列表，可能有多条资讯（解决）
    print('----开始----')
    with open(
            os.path.join('news_data_from2016to2021\companies',
                         str(year) + '_data_new_.json'), 'wb+') as json_file:
        for index in range(1):  # 先拿一天做实验试试  
            news_dict = OrderedDict()
            is_open = trade_day_data.loc[index, 'is_open']  #获取该日期是否开市
            news_dict['is_open'] = is_open
            company_data = OrderedDict()
            new_data = news_data.iloc[next_start:, :]
            for i in range(len(company_code)):
                company_dict = company_detail(
                    company_name[i], company_code[i],
                    company_short[i])  #为了降低文件大小，这里就不再计入公司信息了
                exists_news, news_title, news_content, news_classification, end_line = single_company_news(
                    company_name[i], company_code[i], company_short[i],
                    trade_day_data.loc[index, 'date'], new_data)
                company_dict['exists_news'] = exists_news
                company_dict['title'] = news_title
                company_dict['content'] = news_content
                company_dict['classification'] = news_classification
                company_data[company_code[i]] = company_dict
            #但是好像没有明显加快
            #为了加快速度,每一个日期在用完后，就drop掉，这样可能会提高一些速度
            #news_data.drop(index=news_data.loc[news_data['DeclareDate'] ==
            #                                   trade_day_data.loc[index,
            #                                                      'date']].index)
            #index要进行重置
            #news_data = news_data.reset_index(drop=True)
            next_start = end_line
            news_dict['company'] = company_data
            total_dict[trade_day_data.loc[index, 'date']] = news_dict
            json_str = json.dumps(total_dict, indent=4,
                                  ensure_ascii=False).encode(encoding='utf-8')
            json_file.write(json_str)
            print('完成{}的处理！'.format(trade_day_data.loc[index, 'date']))
            total_dict.pop(trade_day_data.loc[index, 'date'])
        json_file.close()
    print('----完成----')


def maintain_five_year(year_list, company_final, trade_date_path):
    """将五年的数据进行分类保存

    Args:
        year_list (list): 年份的列表，[2020,2021]这种
        company_final (str): 路径
        trade_date_path (str): 交易数据路径
    """
    for year in year_list:
        maintain_relevent_news_single_year_a(year, company_final,
                                             trade_date_path)
        print('完成一年的数据分类')


###########################################################################################################################


#该函数由于数据量过大实际上不好进行拼接操作，后面这几个函数暂时没用，有一些问题
def txt_concate():
    """
    将几年的新闻资讯数据进行汇总,之后方便进行统一的操作,比如可视化和数据清洗.
    """
    news_total = pd.DataFrame(columns=columns_name)
    #注意由于内存大小限制，没法全部汇总只能分来两个，最后手动复制过去
    for year in year_list:
        news_name = os.path.join(news_path, year, file_name)
        start_time = time.time()
        news_data = pd.read_excel(news_name)
        end_time = time.time()
        print('读取' + year + '文件花费的时间为' + str(end_time - start_time))
        news2use = news_data[columns_name]
        news_total = pd.concat([news_total, news2use])
    store_time = time.time()
    news_total.to_csv('all_news.txt',
                      sep='\t',
                      index=False,
                      header=columns_name)
    print('数据存储花费的时间为' + str(time.time() - store_time))
    print('完成!')


def cut_sent(para):
    """中文分句处理，去掉网址链接

    Args:
        para (string): 中文文章或者段落

    Returns:
        [list]: 分句后组成的列表，列表元素为中文的句子
    """
    hanzi_punctuation_ch = hanzi.punctuation
    string_punctuation_en = string.punctuation
    re_punctuation_ch1 = "^([{}]+)".format(hanzi_punctuation_ch)
    re_punctuation_ch2 = "^《[{}]+》".format(hanzi_punctuation_ch)
    re_punctuation_en = "^[{}]+".format(string_punctuation_en)
    para = re.sub('^([http|https]://([\w-]+\.)+[\w-]+(/[\w-./?%&=]*)?$)', r"",
                  para)  # 网址
    para = re.sub(re_punctuation_ch1, r'', para)
    para = re.sub(re_punctuation_ch2, r'', para)
    para = re.sub(re_punctuation_en, r'', para)
    para = re.sub('^([\u4e00-\u9fa5\w]+)', r"", para)  # 圆括号括号内有汉字
    para = re.sub('^《[\u4e00-\u9fa5\w]+》', r"", para)  # 书名号内有汉字
    para = re.sub('摘要', r'', para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def remove_useless_sentence():
    """
    依据是否含有无意义的单词判断整个句子的情况。
    """
    useless_word = ['']
    for year in year_list:
        news_name = os.path.join(news_path, year, file_name)
        news_data = pd.read_excel(news_name)
        print('已读取' + year + '年新闻数据！')
        news2use = news_data[columns_name]
        #news_content = news2use['NewsContent']
        for idx in range(len(news2use)):
            sentence_list = re.findall(hanzi.sentence,
                                       str(news2use.loc[idx, 'NewsContent']))
            left_sentence = [
                sentence if useless_word[-1] not in sentence else ''
                for sentence in sentence_list
            ]

        print('完成新闻内容的修改')

        #news2use['NewsContent'] = news_content
        #news2use.to_csv(os.path.join(news_path,'companies',year+'.txt'), sep=' ', index=False,header=False,quoting=csv.QUOTE_NONE,escapechar=' ')
        print('成功保存一次数据!')


#######################################################################################################################

if __name__ == '__main__':
    #txt_concate()
    #remove_useless_sentence()
    #trade_day_data = trade_day(trade_date_path)
    #print(trade_day_data)
    #print(get_company_name_code())
    #year_list = [2016, 2017, 2018, 2019, 2020, 2021]
    year_list = [2017]
    maintain_five_year(year_list, company_final, trade_date_path)
