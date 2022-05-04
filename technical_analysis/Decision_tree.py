import pandas as pd
import os
import numpy as np
from random import choice
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sys

sys.path.append(".")
from preprocessing.price_preprocess import single_company_price_2tag, new_left_company, missing_fulfill, get_trade_day
from utils_ml import process_x, train_pre, classification_result, plot_ROC

#储存最后使用的公司的路径
company_final = 'D:\毕设code\\news_data_from2016to2021\companies\\final_use'
#交易日数据
trade_cal = 'D:\毕设code\\news_data_from2016to2021\companies\\trade_cal_clean.csv'
#将roc曲线保存的路径
figure_path = 'D:\毕设code\\technical_model_result\\figure'


###基于历史数据进行预测,先获得数据集,以下都是基于已经确定了特征之后的处理
def single_company_data(company_code, path, columns):
    """获取指定列的数据,作为特征x。
    (注,这个地方的实施方式类似于single_company_price_2tag,没有直接调用,主要是考虑到一开始的设计的功能就不一致,
    所以为了一致就没有改动。)

    Args:
        company_code (str): _description_
        path (str): 交易日数据的路径,指的是clean截尾的文件
        colums (list): 需要考虑的特征的list

    Returns:
        dataframe: 所需数据的dataframe
    """
    company_hist = pd.read_csv(os.path.join(company_final,
                                            company_code + '.csv'),
                               encoding='gbk').iloc[:, 2:-1]
    company_hist = company_hist[
        company_hist['trade_date'] >= 20170101].reset_index(drop=True)
    trade_day_data = get_trade_day(path)  #也是日期在2017年7月一日之后的数据
    company_hist = missing_fulfill(trade_day_data, company_hist)
    return company_hist[columns]


def get_x_y_data(company_list, path, columns, processing_type):
    """返回划分好的训练集和测试集,目前不考虑验证集(validation set)

    Args:
        company_code (list): _description_
        path (str): _description_
        columns (str): _description_
        processing_type (str): 预处理的类型

    Returns:
        ndarray: 划分好的训练集和测试集的元组
    """
    #下面将所有公司的数据拼接在一起
    if not os.path.exists('technical_analysis\\x_data.npy') and not os.path.exists('technical_analysis\\y_data.npy'):
        print('不存在缓存文件...')
        y_list = []
        x_list = []
        for company_code in company_list:
            y_list.append(single_company_price_2tag(company_code, trade_cal))

            x_list.append(
                single_company_data(company_code, path,
                                    columns).iloc[:-1, :].values)

        x, y = 0, 0
        for index, (elem_x, elem_y) in enumerate(zip(x_list, y_list)):
            if index == 0:
                x, y = process_x(processing_type, elem_x), elem_y
            else:
                x = np.concatenate((x, process_x(processing_type, elem_x)),
                                   axis=0)
                y = np.concatenate((y, elem_y))  #没有多余维度所以不需要axis
        print('保存缓存文件...')
        with open('x_data.npy', 'wb') as f:
            np.save(f, x)
        with open('y_data.npy', 'wb') as f:
            np.save(f, y)
        print('保存成功...')
    else:
        x = np.load('technical_analysis\\x_data.npy')
        y = np.load('technical_analysis\\y_data.npy')

    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=1234)
    return train_x, test_x, train_y, test_y


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    decisiontree = DecisionTreeClassifier(min_samples_leaf=1,min_samples_split=3,random_state=1234)  #实例化
    #参数设置
    param_grid = {
        'max_depth': np.arange(29, 40),
        #'min_samples_leaf': np.arange(1, 8), #1
        #'min_samples_split': np.arange(2, 8) #3
    }
    #设置10折进行交叉验证
    model = GridSearchCV(decisiontree, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    cm_type = ['train_result', 'test_result']
    #确定公司代码
    print('确定公司ing...')
    company_list = new_left_company(company_final)
    company_list = list(map(lambda x: x.replace('.csv', ''), company_list))
    #company_use = choice(company_list)
    #获取数据集
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = get_x_y_data(
        #company_use.replace('.csv', ''),
        company_list,
        trade_cal,
        columns=['open', 'high', 'low', 'pre_close'],
        processing_type='minmax')
    #获取模型
    print('获取模型ing...')
    model = model_design()
    #开始训练
    print('开始训练ing...')
    confusion_matrix, train_result, test_result = train_pre(
        model, train_x, test_x, train_y, test_y)
    #保存模型
    print('保存模型ing...')
    classification_result(confusion_matrix, cm_type, train_result, test_result,
                          'DecisionTree')
    #roc曲线可视化
    print('可视化ing...')
    for index, lable in enumerate([train_result, test_result]):
        if index == 0 and train_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path,
                                           'DecisionTree_train_roc.jpg'))
        elif index == 1 and test_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path,
                                           'DecisionTree_test_roc.jpg'))
        else:
            print('模型没法预测分类概率，没法进行可视化')
            continue
    print('全部结束！！！！')


if __name__ == '__main__':
    main()