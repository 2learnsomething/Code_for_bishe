# -- coding:utf-8 --

from cProfile import label
from cmath import nan
from importlib.resources import path
from pickle import TRUE
import time
from tkinter import X
from tracemalloc import start
import pandas as pd
import numpy as np
from numpy import place, random
import datetime
import traceback
import shutil
import os
import matplotlib.pyplot as plt
from random import choice
import tushare as ts
import warnings

warnings.filterwarnings('ignore')

#np.random.seed(1234)

# 获得我们所研究的公司列表以及代码，主要是深交所上市的2016年前上市的公司
company_path = 'news_data_from2016to2021\companies\深交所A股列表_主板.xlsx'

#基于tushare接口的数据。
ts_company_path = 'news_data_from2016to2021\companies\\ts_stock_price'

# 各个公司的过去股价信息,需要确定具体的股票代码，网易的数据
company_price = 'news_data_from2016to2021\companies\stock_price'

# 交易日数据
trade_day = 'news_data_from2016to2021\companies'

# tushare 的登录token,需要自己注册,记得删掉自己的
ts_entry = "0eab484a2cff67f1cab83d7154a3e74a3accbf0f9d18a05839839dcf"
#储存最后使用的公司的路径
company_final = 'D:\毕设code\\news_data_from2016to2021\companies\\final_use'


def get_trade_cal():
    """
    获取深交所的交易日信息,并不是一年中每天都有交易数据,所以需要交易日数据,用来判断。
    """
    pro = ts.pro_api(ts_entry)
    trade_cal = pro.trade_cal(exchange='SZSE',
                              start_date='20160601',
                              end_date='20211231')
    print('Total trade days from 20160601 to 20211231')
    save_dir = trade_day + '\\' + 'trade_cal.csv'
    trade_cal.to_csv(save_dir)


def get_stock_price():
    """
    从tushare接口获取数据，主要是发现从网易获得的数据有空缺，比如中间某段时间数据的空缺
    """
    pro = ts.pro_api(ts_entry)
    ts.set_token(ts_entry)
    company_name = get_company_name_code()
    startdate = '20160601'
    enddate = '20211231'
    num = 1
    for elem in company_name:
        if os.path.exists(ts_company_path + '\\' + elem[1] + '.csv'):
            print('processed')
            continue
        else:
            price_data = pro.daily(ts_code=elem[1] + '.SZ',
                                   start_date=startdate,
                                   end_date=enddate)
            price_data.to_csv(ts_company_path + '\\' + elem[1] + '.csv')
            #停止一秒
            time.sleep(1.5)
        print(f'Processing no.{num} file !')
        num += 1


def get_company_name_code():
    """获得公司全称,代码以及简称等信息

    Args:
        company_path (str): 文件路径,之后也都将以相对路径为主

    Returns:
        list: 全称,代码,简称组成的元组的列表
    """
    company_detail = pd.read_excel(company_path)
    name_code = company_detail.loc[:, ['公司全称', 'A股代码', 'A股简称']]
    #代码部分是数字所以还需要做一些处理,使用内置的zfill函数
    name_code['A股代码'] = pd.DataFrame(
        [str(code).zfill(6) for code in name_code['A股代码'].values],
        columns=['A股代码'])
    #股票简称支付中间存在空格
    name_code['A股简称'] = pd.DataFrame(
        [name.replace(' ', '') for name in name_code['A股简称'].values],
        columns=['A股简称'])

    return zip(name_code['公司全称'].values, name_code['A股代码'].values,
               name_code['A股简称'])


def date_trans(date):
    """时间格式转换

    Args:
        date (str): 字符串'20210112'这种

    Returns:
        [time]: 规范化之后的时间格式,2021-01-12
    """
    date = pd.to_datetime(date)
    return date.strftime('%Y-%m-%d')


def stock_price_date():
    """获取不同公司的数据量，包括最早的股价数据和最晚的股价数据日期，以及数据量

    Returns:
        [tuple]: 数据量多少的列表，最早时间和最晚时间列表,以及股票代码
    """
    #获取每家公司最早的数据日期和最晚的日期,以及数据数目,开始和结束顺序有点问题，不过不大
    company_name = get_company_name_code()
    start_price_date = []  #最近的数据日期
    end_price_date = []  #最早的数据日期
    data_num = []
    company_list = []
    for elem in company_name:
        path = ts_company_path + '\\' + elem[1] + '.csv'
        if os.path.exists(path):
            company_price = pd.read_csv(path)
            start_date = str(company_price.loc[0, 'trade_date'])
            end_date = str(company_price.loc[len(company_price) - 1,
                                             'trade_date'])
            start_price_date.append(date_trans(start_date))
            end_price_date.append(date_trans(end_date))
            data_num.append(len(company_price))
            company_list.append(elem[1])
        else:
            print('file not found!')
            continue
    return data_num, start_price_date, end_price_date, company_list


def datetime_trans(time):
    """生成用于判断的时间格式见 https://www.jb51.net/article/147429.htm

    Args:
        time (str): 日期字符串，Y-m-d

    Returns:
        [datetime]: datetime的时间格式
    """
    time_detail = time.split('-')
    return datetime.date(int(time_detail[0]), int(time_detail[1]),
                         int(time_detail[2]))


def move_file(src_path, dst_path, file):
    """文件移动

    Args:
        src_path (str): 源路径名
        dst_path (str): 目标路径名
        file (str): 文件名
    """
    try:
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        traceback.print_exc()


def remove_less_data_com():
    """挑选出数据量较多的公司

    Returns:
        [list]: 留下来用于最后实验的公司代码
    """
    end_date = date_trans('20170101')
    start_date = date_trans('20211231')
    company_left = []  #将用于最后实验的公司代码
    date_left = []
    data_num, start_price_date, end_price_date, company_list = stock_price_date(
    )
    num_left = 0
    for i in range(len(company_list)):
        if start_date == start_price_date[i] and datetime_trans(
                end_date).__ge__(datetime_trans(end_price_date[i])):
            num_left += 1
            company_left.append(company_list[i])
            print('code: ' + str(company_list[i]) + 'is left')
            move_file(ts_company_path, company_final, company_list[i] + '.csv')
            print("成功移动一个文件!")
        else:
            print('该公司的最早日期为' + end_price_date[i])
            print('该公司的最晚日期为' + start_price_date[i])
            print('该公司的数据量为' + str(data_num[i]))
            date_left.append(data_num[i])
    print('最终留下来的公司数目为' + str(num_left))
    return company_left


def new_left_company(src_path):
    """返回文件列表

    Args:
        src_path (str): 文件父目录

    Returns:
        [list]: 文件列表
    """
    return os.listdir(src_path)


def single_company_data_info(src_path):
    """返回数据的简单统计特征

    Returns:
        [dataframe]: 数据的统计特征,包括极值,方差等。
    """
    #随机确定公司
    company_name_code = new_left_company(src_path)
    company_code = choice(company_name_code)
    #读数据
    company_hist = pd.read_csv(src_path + '\\' + company_code, encoding='gbk')
    return company_hist.describe()


def price_visualization(src_path):
    """各公司股价走势比较

    Args:
        src_path (str): 公式文件路径，这里选取的是最后留下来的公式数据。
    """
    #获取股票代码
    company_name_code = get_company_name_code()
    company_list = []
    company_name = []
    left_company = new_left_company(src_path)
    left_company = list(map(lambda x: x.replace('.csv', ''), left_company))
    for elem in company_name_code:
        if elem[1] in left_company:
            company_list.append(elem[1])
            company_name.append(elem[2])
        else:
            continue

    #这里就以代码靠前的七家公司为例
    company_to_show_list = company_list[:7]
    company_name_list = company_name[:7]
    #读取数据
    company_hist1 = pd.read_csv(src_path + '\\' + company_to_show_list[0] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist2 = pd.read_csv(src_path + '\\' + company_to_show_list[1] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist3 = pd.read_csv(src_path + '\\' + company_to_show_list[2] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist4 = pd.read_csv(src_path + '\\' + company_to_show_list[3] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist5 = pd.read_csv(src_path + '\\' + company_to_show_list[4] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist6 = pd.read_csv(src_path + '\\' + company_to_show_list[5] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    company_hist7 = pd.read_csv(src_path + '\\' + company_to_show_list[6] +
                                '.csv',
                                encoding='gbk').iloc[:, 1:]
    #获取时间
    date = company_hist1['trade_date']
    #指定需要的数据,一般五项
    k_data_column = ['open', 'high', 'low', 'close', 'vol']
    #获取数据
    k_company_data1 = company_hist1[k_data_column]
    k_company_data2 = company_hist2[k_data_column]
    k_company_data3 = company_hist3[k_data_column]
    k_company_data4 = company_hist4[k_data_column]
    k_company_data5 = company_hist5[k_data_column]
    k_company_data6 = company_hist6[k_data_column]
    k_company_data7 = company_hist7[k_data_column]
    company_close1 = k_company_data1['close']
    company_close2 = k_company_data2['close']
    company_close3 = k_company_data3['close']
    company_close4 = k_company_data4['close']
    company_close5 = k_company_data5['close']
    company_close6 = k_company_data6['close']
    company_close7 = k_company_data7['close']
    company_close_total = pd.concat([
        date, company_close1, company_close2, company_close3, company_close4,
        company_close5, company_close6, company_close7
    ],
                                    axis=1)
    #k_company_data1.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data2.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data3.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data4.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data5.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data6.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    #k_company_data7.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    columns = ['日期']
    for name in company_name_list:
        columns.append(name)
    print(columns)
    company_close_total.columns = columns
    company_close_total['日期'] = company_close_total['日期'].apply(
        lambda _: str(_))
    company_close_total['日期'] = pd.to_datetime(company_close_total['日期'])
    company_close_total.set_index('日期', inplace=True)
    #
    #横轴x是股票时间，默认就可以了
    #纵轴y是收盘价Close这一列数据
    #plot默认是线条图
    #使用label自定义图例

    #设置画板大小
    plt.figure(figsize=(15, 12))
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.plot(company_close_total, label=columns[1:])
    #x坐标轴文本
    plt.xlabel('时间')
    #y坐标轴文本
    plt.ylabel('收盘价')
    #图片标题
    plt.title('2016年6月一日-2021年12月31日期间几家公司股票收盘价走势')
    #显示网格
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
    print(company_close_total)


#事实上存在数据缺失的情况，这个需要解决
def get_trade_day(path):
    """获取交易日数据,注意应该是2017-01-01 到 2021-12-31之间，之后的标签也是基于这中间的数据进行的

    Args:
        path (str): 交易日数据

    Returns:
        [dataframe]: 交易日的dataframe,只有两列
    """
    date = pd.read_csv(path, encoding='gbk')
    date = date[date['cal_date'] >= 20170101].reset_index(drop=True)
    date = date[date['is_open'] == 1]
    return date.reset_index(drop=True)


def insert(df, i, df_add):
    """在指定行插入数据

    Args:
        df (dataframe): 要操作的数据
        i (int): 行数
        df_add (dataframe): 要插入的数据

    Returns:
        [dataframe]: 插入之后的dataframe
    """
    # 指定第i行插入一行数据
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=True)
    return df_new


def missing_fulfill(trade_date, company_price):
    """缺失值补充

    Args:
        trade_date (dataframe): 交易日数据的dataframe
        company_price (dataframe): 公司数据的dataframe

    Returns:
        dataframe: 补充缺失值之后的数据
    """
    total_num = len(trade_date)
    company_data_num = len(company_price)
    if total_num == company_data_num:
        return company_price
    else:
        data_need = total_num - company_data_num
        print('需要补充的数据量为：',data_need)
        if data_need > 0:
            flag = True
            temp = 0  #记录哪一行有问题
            times = 0  #记录修改了多少次
            while flag:
                for i in range(temp, len(company_price)):
                    if company_price.loc[i, 'trade_date'] == trade_date.loc[
                            total_num - 1 - i, 'cal_date']:
                        continue  #说明在trade_date.loc[-i,'cal_date']日期有数据
                    else:
                        value = [trade_date.loc[total_num - i, 'cal_date']
                                 ] + [nan] * (len(company_price.columns) - 1)
                        df_add = dict(zip(company_price.columns, value))
                        df_add = pd.DataFrame([df_add])
                        company_price = insert(company_price, i, df_add)
                        temp = i
                        times += 1
                        break
                if times == data_need:
                    flag = False
                else:
                    flag = True
            #这里采取线条插值，阶数定位3
            company_price = company_price.interpolate(method='spline', order=3)
            print('trade days:' + str(total_num))
            print('company data number:' + str(len(company_price)))
            return company_price
        else:
            print('公司数据量更多，请检查具体数据内容！')


def single_company_price_2tag(company_code, path):
    """基于收盘价生成股价浮动的标签

    Args:
        company_code (str): 公司编码

    Returns:
        [list]: 返回股价上升或者下降的tag,1代表上升,否则就是下降
    """
    #准备存放标签，是上升还是下降，先采取二分类
    company_hist = pd.read_csv(os.path.join(company_final,
                                            company_code + '.csv'),
                               encoding='gbk').iloc[:, 2:-1]
    company_hist = company_hist[
        company_hist['trade_date'] >= 20170101].reset_index(drop=True)
    trade_day_data = get_trade_day(path)  #也是日期在2017年7月一日之后的数据
    company_hist = missing_fulfill(trade_day_data, company_hist)  #进行缺失数据的处理
    # 该标签都是基于收盘价得到的
    close_price = np.squeeze(company_hist['close'].values)
    price_gap = close_price[1:] - close_price[:-1]
    price_gap[np.where(price_gap >= 0)] = 1
    price_gap[np.where(price_gap < 0)] = 0
    return price_gap


def single_company_price_3tag(company_code, path):
    """基于收盘价生成股价浮动的标签

    Args:
        company_code (str): 公司编码

    Returns:
        [list]: 返回股价上升或者下降的tag,1代表上升,-1代表下降
    """
    #准备存放标签，是上升还是下降，先采取二分类
    company_hist = pd.read_csv(os.path.join(company_final,
                                            company_code + '.csv'),
                               encoding='gbk').iloc[:, 2:-1]
    company_hist = company_hist[
        company_hist['trade_date'] >= 20170101].reset_index(drop=True)
    trade_day = get_trade_day(path)  #也是日期在2017年7月一日之后的数据
    company_hist = missing_fulfill(trade_day, company_hist)  #进行缺失数据的处理
    # 该标签都是基于收盘价得到的
    price_gap = company_hist['close']
    close_price = np.squeeze(company_hist['close'].values)
    price_gap = (close_price[1:] - close_price[:-1]) / close_price[:-1]
    price_gap[np.where(price_gap > 0.01)] = 1
    price_gap[np.where(price_gap < -0.01)] = -1
    mask_more = (price_gap > 0.01).tolist()
    mask_less = (price_gap < -0.01).tolist()
    mask_rest = [
        False if mask_more[i] or mask_less[i] else True
        for i in range(len(price_gap))
    ]
    price_gap[mask_rest] = 0
    #np.savetxt(trade_day + '\\' + 'mask', X=price_gap)
    #price_gap[np.where(price_gap <= 0.025 and price_gap >= -0.025)] = 0,这个没法运行
    #print((price_gap >= -0.025) and (price_gap <= 0.025))
    return price_gap


def total_company_price_2tag(path):
    """将所有的公司股价信息进行汇总,为一个字典

    Returns:
        json: 将标签信息保存json文件
    """
    tag_dic = {}
    company_name_code = new_left_company(company_final)
    for elem in company_name_code:
        tag_dic[elem[2]] = single_company_price_2tag(elem[2], path)
    #也可以保存
    #with open(os.path.join(news_data_from2016to2021\companies,'tag_dic.json'),'w') as f:
    #   json.dumpy(tag_dic)
    return tag_dic


def total_company_price_3tag(path):
    """将所有的公司股价信息进行汇总,为一个字典

    Returns:
        json: 将标签信息保存json文件
    """
    tag_dic = {}
    company_name_code = new_left_company(company_final)
    for elem in company_name_code:
        tag_dic[elem[2]] = single_company_price_3tag(elem[2], path)
    #也可以保存
    #with open(os.path.join(news_data_from2016to2021\companies,'tag_dic.json'),'w') as f:
    #   json.dumpy(tag_dic)
    return tag_dic


if __name__ == '__main__':
    #price_tag = single_company_price_3tag('000001')
    #print(price_tag)
    #print(single_company_data_info())
    #get_stock_price()
    #price_visualization()
    #left = remove_less_data_com()
    #print(left)
    #single_company_data_info(company_final)
    # price_visualization(company_final)
    date = get_trade_day(trade_day + '\\' + 'trade_cal_clean.csv')
    company_hist = pd.read_csv(os.path.join(company_final, '002137' + '.csv'),
                               encoding='gbk').iloc[:, 2:-1]
    company_hist = company_hist[
        company_hist['trade_date'] >= 20170101].reset_index(drop=True)
    print(company_hist.shape)
    price_updown =  missing_fulfill(date, company_hist)
    print(price_updown.shape)
