from cProfile import label
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from price_preprocess import new_left_company, date_trans, get_company_name_code

# 获得我们所研究的公司列表以及代码，主要是深交所上市的2016年前上市的公司
company_path = 'news_data_from2016to2021\companies\深交所A股列表_主板.xlsx'
#储存最后使用的公司的路径
company_final = 'D:\毕设code\\news_data_from2016to2021\companies\\final_use'

#详情参考https://zhuanlan.zhihu.com/p/260265201


#该函数做了简易修改，添加了一个变量
def get_company_name_code(company_path):
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


def company_list():
    """留下的公司的代码信息

    Returns:
        list: 公司代码的列表
    """
    company_ = new_left_company(company_final)
    return list(map(lambda x: x.replace('.csv', ''), company_))


def company_name():
    """获取留下的公司的简称，代码等信息

    Returns:
        tuple: 公司信息的三元组
    """
    left_company = company_list()
    new_left_company_list = []
    company_name_code = get_company_name_code(company_path)
    for elem in company_name_code:
        if elem[1] in left_company:
            new_left_company_list.append(elem)
        else:
            continue
    return new_left_company_list


def get_random_company():
    """随即返回某家公司的数据

    Returns:
        dataframe: 股价数据的dataframe
    """
    new_left_company_list = company_name()
    #随机选取一个公司
    company_code = choice(new_left_company_list)
    company_data = pd.read_csv(
        os.path.join(company_final, company_code[1] + '.csv')).iloc[:, 2:-1]
    company_data['trade_date'] = company_data['trade_date'].apply(
        lambda x: date_trans(str(x)))
    return company_data, company_code


# 单只股票价格折线图
def single_stock_line(company_data, company_code):
    """绘制折线图
    """
    #company_data, company_code = get_random_company()
    plot_data = company_data.loc[:, ['trade_date', 'close']]
    plot_data = plot_data.iloc[::-1].reset_index(drop=True)
    plot_data.columns = ['交易日日期', '收盘价']
    plot_data.set_index('交易日日期').plot(
        kind='line',
        figsize=(12, 6),
        legend=True,
        label=company_code[2],
        title=f'2016年6月1日至2021年12月31日期间{company_code[2]}收盘价格趋势图')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('交易日日期')
    plt.ylabel('收盘价')
    plt.show()


#股价均线图
def single_average_line(company_data, company_code):
    """单个公司的几种折现走势对比图
    """
    #company_data, company_code = get_random_company()
    company_data['thirty_ma'] = company_data.close.rolling(window=30).mean()
    company_data['sisty_ma'] = company_data.close.rolling(window=60).mean()
    company_data['ninety_ma'] = company_data.close.rolling(window=90).mean()
    fig, ax = plt.subplots(figsize=(12, 9))
    # 使用循环生成４条折线
    lst_col = ['close', 'thirty_ma', 'sisty_ma', 'ninety_ma']
    for c in lst_col:
        plot_data = company_data.loc[:, ['trade_date', c]]
        plot_data = plot_data.iloc[::-1].reset_index(drop=True)
        if lst_col.index(c) == 0:
            plot_data.columns = ['交易日日期', '收盘价']
        elif lst_col.index(c) == 1:
            plot_data.columns = ['交易日日期', '30日线']
        elif lst_col.index(c) == 2:
            plot_data.columns = ['交易日日期', '60日线']
        elif lst_col.index(c) == 3:
            plot_data.columns = ['交易日日期', '90日线']
        plot_data.set_index('交易日日期').plot(
            kind='line',
            ax=ax,
            title=f'2016年6月1日至2021年12月31日期间{company_code[2]}股价趋势图')
    # 更新标签
    lst_led = ['日收盘价', '30日线', '60日线', '90日线']
    ax.legend(lst_led)
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('交易日日期')
    plt.ylabel('股价')
    plt.show()


def get_year_data(year, data):
    """返回指定年份的数据,为了最后的显示做出一些小的让步

    Args:
        year (int): 年份
        data (dataframe): 数据

    Returns:
        dataframe: 指定年份的数据
    """
    data['year'] = data['trade_date'].apply(lambda x: x.split('-')[0])
    data['month'] = data['trade_date'].apply(lambda x: x.split('-')[1])
    data['day'] = data['trade_date'].apply(lambda x: x.split('-')[2])
    specific_data = data[data['year'].apply(lambda x: int(x)) ==
                         year].reset_index(drop=True)
    #data['年'] = data['trade_date'].apply(lambda x : x.split('-')[0])
    #data['月'] = data['trade_date'].apply(lambda x : x.split('-')[1])
    #data['日'] = data['trade_date'].apply(lambda x : x.split('-')[2])
    #specific_data = data[data['年'].apply(lambda x: int(x)) == year].reset_index(drop=True)
    return specific_data


#月箱线图
def single_stock_box(year, company_data, company_code):
    """指定年份的月箱线图，不如k线图那么好，之后可以考虑添加

    Args:
        year (int): 年份
    """
    # 生成月和日列
    #company_data, company_code = get_random_company()
    company_data = get_year_data(year, company_data)
    company_data.loc[:,
                     ['month', 'close']].boxplot(by='month',
                                                 figsize=(12,
                                                          9)).set(xlabel=None)
    plt.title(f'{company_code[2]}的{year}年月k图')
    plt.xlabel('月份')
    plt.ylabel('股价')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.show()


#月平均成交量柱状图
def single_vol_histogram(year, company_data, company_code):
    """指定年份的月平均成交量柱线图

    Args:
        year (int): 年份
    """
    #company_data, company_code = get_random_company()
    company_data = get_year_data(year, company_data)
    company_data.groupby('month')['vol'].mean().plot(kind='bar')
    plt.title(f'{company_code[2]}于{year}年月均交易量图')
    plt.xlabel('月份')
    plt.ylabel('成交量')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.show()


#收盘价与成交量散点图
def single_close_vol_scatter(year, company_data, company_code):
    """某公司收盘价和成交量之间的散点图

    Args:
        year (int): 年份
    """
    #company_data, company_code = get_random_company()
    company_data = get_year_data(year, company_data)
    company_data.loc[:, ['close', 'vol']].plot(
        x='close',
        y='vol',
        kind='scatter',
        figsize=(12, 9),
        color='green',
        title=f'{company_code[2]}于年{year}收盘价与成交量散点图')
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('收盘价')
    plt.ylabel('成交量')
    plt.show()
    #print(f'{company_code[2]}收盘价与成交量相关矩阵')
    #print(company_data.loc[:, ['close', 'vol']].corr)


def main():
    year = 2021
    company_data, company_code = get_random_company()
    single_stock_line(company_data, company_code)
    single_average_line(company_data, company_code)
    single_stock_box(year, company_data, company_code)
    single_vol_histogram(year, company_data, company_code)
    single_close_vol_scatter(year, company_data, company_code)


if __name__ == '__main__':
    #single_stock_line()
    #single_average_line()
    #single_stock_box(2021)
    #single_vol_histogram(2021)
    #single_close_vol_scatter(2021)
    main()
