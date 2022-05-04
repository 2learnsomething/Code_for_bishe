from math import gamma
import os
import sys

sys.path.append('.')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils_ml import process_x, train_pre, classification_result, plot_ROC

#交易日数据
trade_cal = '/new_python_for_gnn/毕设code/news_data_from2016to2021/companies/trade_cal_clean.csv'
#将roc曲线保存的路径
figure_path = '/new_python_for_gnn/毕设code/technical_model_result/figure'


#获取数据
def get_x_y_data():
    """返回划分好的训练集和测试集,目前不考虑验证集(validation set)

    Args:
        company_code (list): _description_
        path (str): _description_
        columns (str): _description_
        processing_type (str): 预处理的类型

    Returns:
        ndarray: 划分好的训练集和测试集的元组
    """
    #由于第一个实现的是decision——tree,已经保存了相关的数据文件，所以这里直接读取，就可以节省数据concanate的时间
    x = np.load('technical_analysis/x_data.npy')
    y = np.load('technical_analysis/y_data.npy')

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
    lr = LogisticRegression(solver='liblinear')  #实例化

    #参数的搜索范围
    penaltys = ['l1', 'l2']
    Cs = [0.1, 1, 10, 100, 1000]
    #调优的参数集合，搜索网格为x5，在网格上的交叉点进行搜索
    param_grid = dict(penalty=penaltys, C=Cs)

    #设置10折进行交叉验证
    model = GridSearchCV(lr,
                         param_grid,
                         cv=10,
                         scoring='neg_log_loss',
                         verbose=2,
                         n_jobs=-1)
    # 进行预测
    return model


def main():
    cm_type = ['train_result', 'test_result']
    #确定公司代码
    #print('确定公司ing...')
    #company_list = new_left_company(company_final)
    #company_list = list(map(lambda x: x.replace('.csv', ''), company_list))
    #company_use = choice(company_list)
    #获取数据集
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = get_x_y_data()
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
                          'LR')
    #roc曲线可视化
    print('可视化ing...')
    for index, lable in enumerate([train_result, test_result]):
        if index == 0 and train_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path, 'LR_train_roc.jpg'))
        elif index == 1 and test_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path, 'LR_test_roc.jpg'))
        else:
            print('模型没法预测分类概率，没法进行可视化')
            continue
    print('全部结束！！！！')


if __name__ == '__main__':
    main()
