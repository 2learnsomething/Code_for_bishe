from turtle import color
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.append('.')
from preprocessing.price_preprocess import new_left_company
from utils import path

path = path.rsplit('\\', 1)[0]


def get_result(data_type='precision'):
    """获取想要可视化的分类结果数据

    Args:
        train_or_test (str, optional): 可视化测试集结果还是训练集结果. Defaults to 'test'.
        data_type (str, optional): 想要可视化的数据. Defaults to 'precision'.

    Returns:
        tuple: 模型名称和数据
    """
    result_list = new_left_company(
        'D:\毕设code\knowledge_graph\dgl_model_result\\test')
    result_list = [elem for elem in result_list if elem.find('.json') != -1]
    print(result_list)
    model_name = list(
        map(lambda x: x.replace('_new.json', ''),
            result_list))  #获取模型的名称
    print(model_name)
    data_list = []
    for file_name in result_list:
        with open(os.path.join('D:\毕设code\knowledge_graph\dgl_model_result\\test',file_name), 'r') as f:
            cont = json.load(f)
        data_list.append(cont[data_type])
    return model_name, data_list


def name_tran(data_type):
    """将指标类型转化为中文

    Args:
        data_type (str): 需要转化的指标类型

    Returns:
        str: 中文字符串
    """
    if data_type == 'precision':
        return '精确率'
    elif data_type == 'accuracy':
        return '准确率'
    elif data_type == 'recall':
        return '召回率'
    elif data_type == 'f1':
        return 'F1得分'
    elif data_type == 'tnr':
        return '特异度'
    elif data_type == 'fpr':
        return '假报警率'


def vis_data(model_name, data_list, data_type):
    """对分类指标进行绘制可视化

    Args:
        model_name (list): 模型列表
        data_list (list): 可视化的数据
        data_type (str): 可视化的指标
    """
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('模型类型')
    data_type = name_tran(data_type)
    plt.ylabel(data_type)
    plt.plot(model_name, data_list, 'r*-.')
    for name, data in zip(model_name, data_list):
        plt.text(name, data, round(data, 2), color='blue', fontsize=8)
    #plt.show()
    plt.savefig(os.path.join('D:\毕设code\knowledge_graph\dgl_model_result\\figure', data_type + '.pdf'))
    print('保存图片成功')


def main():
    data_type_list = ['precision', 'accuracy', 'recall', 'f1', 'tnr', 'fpr']
    for data_type in data_type_list:
        #print(data_type)
        model_name, data_list = get_result(data_type)
        #print(model_name,data_list)
        vis_data(model_name, data_list, data_type)


if __name__ == '__main__':
    main()